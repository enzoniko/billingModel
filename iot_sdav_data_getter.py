import json
import os, sys, requests, base64, zipfile, io, shutil, struct
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


#MY_CERTIFICATE = ['./sdav_cert/sdav.pem', './sdav_cert/sdav.key']
USR = "sdavro"
PSS = "sd4vR@"
log_path = "./get_log.txt"
zip_path = "./simulations"
HOST = "iot.ufsc.br"
get_url = "https://"+HOST+"/api/get.php"
SIGNATURE = 400
DEBUG = True
ego_class = 12

MINUTES_IN_THE_FUTURE_TO_LOOK = 60*12*12

def debug(*args):
    if (DEBUG):
        print(*args)
        
    output = ' '.join(map(str, args))
    output = output + "\n"
    with open(log_path, "a") as log_file:
        log_file.write(output)

def decode_motion_vector(sd):
    decode = base64.b64decode(sd['value'])
    speed = struct.unpack('1f', decode[0:4])[0]
    heading = struct.unpack('1f', decode[4:8])[0]
    yaw_rate = struct.unpack('1f', decode[8:12])[0]
    accel = struct.unpack('1f', decode[12:16])[0]
    id = struct.unpack('Q', decode[16:24])[0]
    motion_vector = {
        'speed': speed,
        'heading': heading,
        'yaw_rate': yaw_rate,
        'accel': accel,
        'ID': id,
    }
    return motion_vector


def generate_unit(length, subtype, type_, multi, si):
    # Initialize the unit value as 0
    _u = 0
    
    # Combine bits into the unit
    _u |= (length & 0xFFFF)           # Bits 0-15
    _u |= (subtype & 0xFF) << 16      # Bits 16-23
    _u |= (type_ & 0x1F) << 24        # Bits 24-28
    _u |= (multi & 0x03) << 29        # Bits 30-29
    _u |= (si & 0x01) << 31           # Bit 31
    
    return _u

def get_series(unit, t0, tf, dev, sig):
    global session

    get_series = { "series": {
            "version"    : "1.2",
            "unit"       : unit,
            "t0"         : t0,
            "t1"         : tf,
            "x"          : 0,
            "y"          : 0,
            "z"          : 0,
            "r"          : 999999999,
            "workflow"   : 0,
            "dev"        : dev, # Unlike version 1.1, dev must be defined in series.  If there is another device with same unit, another series must be created.
            "signature"  : sig
        },
        "credentials" : {
            "domain" : "sdav",
            "username": USR,
            "password": PSS
        }
    }
    debug(get_series)
    session = requests.Session()

    # Create the series
    session.headers = {'Content-type' : 'application/json'}
    #session.cert = MY_CERTIFICATE
    response = session.post(get_url, json.dumps(get_series), verify=False)
    resp = response.status_code
    if resp != 200:
        print(resp, "Get failed, aborting test...")
        return False, ""
    else:
        smartdata = json.loads(response.text)
        if (len(smartdata["series"]) > 0):
            while(True):
                print("Get V1.2 - OK ", len(smartdata["series"]), " SmartData returned, will try to get from")
                get_series["series"]["t0"] = smartdata["series"][-1]["timestamp"]+1
                response = session.post(get_url, json.dumps(get_series), verify=False)
                if (response.status_code == 200):
                    new_smartdata = json.loads(response.text)
                    if (len(new_smartdata["series"]) > 0):
                        smartdata["series"] += new_smartdata["series"]
                    else:
                        break
        return True, smartdata

def get_blob_data_and_save_in_dir(reading, sensor):
    t0 = int(reading["t0"])
    tf = int(reading["tf"])
    if (not os.path.exists(zip_path+"/data/"+sensor["kind"])):
            os.mkdir(zip_path+"/data/"+sensor["kind"])

    debug("\tGetting data in steps of 10 to avoid too big request...")
    aux_t0 = t0
    step = 1000000
    for t1 in  range(t0+step, tf, step): # lets do 10 samples per request
        debug("\tfrom", aux_t0, " to ", t1)
        #resp = get_series(int(r["unit"]), aux_t0, t1, int(r["dev"]), int(r["signature"]))[1]
        resp = get_series(int(reading["unit"]), aux_t0, t1, 0, int(sig))[1] # fixed dev to 0
        debug("\tgot", len(resp["series"]), " SmartData! Writing individual files to:", zip_path+"/data/"+sensor["kind"]+"/")
        for sd in resp["series"]:
            with open(zip_path+"/data/"+sensor["kind"]+"/"+reading["measurement"]+"-"+str(sd["timestamp"])+"_"+str(sig)+".raw", "wb") as f:
                data_bin = base64.b64decode(sd["value"])
                f.write(data_bin)
        debug("\tdone!")
        aux_t0 = t1

def process_zips_smartdata(commit_id, resp, size_resp):
    context = ""
    i = 0
    for i, sd in enumerate(resp["series"]):
        debug("zip", i+1, "/", size_resp)
        try:
            zip_bin = base64.b64decode(sd["value"])
        except Exception as e:
            debug(e, "already used previously?")
            continue
        memfile = io.BytesIO(zip_bin)
        try:
            debug("\textracting zip to:", zip_path)
            with zipfile.ZipFile(memfile, 'r') as zip_ref:
                zip_ref.extractall(zip_path)
            debug("\tdone! loading context.json")
            with open(zip_path+"/context.json") as f:
                _context = json.loads(f.read())
            debug("\tdone!")

            if (not _context["id"].split("\n")[0] == commit_id):
                debug("\tcommit_id_found = ", _context["id"], ", expected=", commit_id, ", deleting content...")
                for filename in os.listdir(zip_path):
                    debug("removing:", filename)
                    file_path = os.path.join(zip_path, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print('Failed to delete %s. Reason: %s' % (file_path, e))
                _context = ""
            else:
                if (_context["description"] == "Compilation Failure! Ignore Context Below."):
                    debug("Found, but context says Compilation Failure! Ignore Context Below.")
                else:
                    context = _context
                    debug("\tcommit_id=", commit_id, "found!, will now download data...")
                    break

        except Exception as e:
            debug("\terror:", e)

    if (context == ""):
        debug("\tcommit_id=", commit_id, "not found! Try other timestamps or increasing 'MINUTES_IN_THE_FUTURE_TO_LOOK'...")
        return

    if (not os.path.exists(zip_path+"/data")):
        os.mkdir(zip_path+"/data")

    sigs = []
    if ("sigs" in context):
        sigs = context["sigs"]
    else:
        sigs = [400]

    for sig in sigs:
        for sensor in context["vehicles"][0]["sensors"]:
            debug("Getting data from ", sensor["kind"])
            if (sensor["kind"] == "CAMERA" or sensor["kind"] == "LIDAR" or sensor["kind"] == "KITTI"):
                if (download_cam_lidar == 1):
                    if (sensor["kind"] == "KITTI"):
                        for r in sensor["readings"]:
                            get_blob_data_and_save_in_dir(r, sensor)
                    else:
                        get_blob_data_and_save_in_dir(sensor["readings"][0], sensor)
            else:
                for r in sensor["readings"]:
                    if (sensor["kind"] == "SUMO.11.1.PHEMlight"):
                        unit = 3834820900
                    else:
                        unit = int(r["unit"])
                    debug("\tgetting ", r["measurement"])
                    resp = get_series(unit, int(r["t0"]), int(r["tf"]), int(r["dev"]), int(sig))[1]
                    debug("\tgot", len(resp["series"]), " SmartData! Writing json to:", zip_path+"/data/"+sensor["kind"]+"_"+r["measurement"]+".json")
                    with open(zip_path+"/data/"+sensor["kind"]+"_"+r["measurement"]+"_"+str(sig)+".json", "w") as f:
                        f.write(json.dumps(resp, indent=4))

        if ("actuation" in context["vehicles"][0]):
            for a in context["vehicles"][0]["actuation"]:
                debug("\tgetting ", a["measurement"])
                resp = get_series(int(a["unit"]), int(a["t0"]), int(a["tf"]), int(a["dev"]), int(sig))[1]
                debug("\tgot", len(resp["series"]), " SmartData! Writing json to:", zip_path+"/data/"+a["measurement"]+".json")
                with open(zip_path+"/data/"+a["measurement"]+"_"+str(sig)+".json", "w") as f:
                    f.write(json.dumps(resp, indent=4))

        if ("telemetry" in context["vehicles"][0]):
            for t in context["vehicles"][0]["telemetry"]:
                debug("\tgetting ", t["measurement"])
                resp = get_series(int(t["unit"]), int(t["t0"]), int(t["tf"]), int(t["dev"]), int(sig))[1]
                debug("\tgot", len(resp["series"]), " SmartData! Writing json to:", zip_path+"/data/"+t["measurement"]+".json")
                with open(zip_path+"/data/"+t["measurement"]+"_"+str(sig)+".json", "w") as f:
                    f.write(json.dumps(resp, indent=4))

        if ("motion_vectors" in context["vehicles"][0]):
            for m in context["vehicles"][0]["motion_vectors"]:
                debug("\tgetting ", m["measurement"])
                if (not m["available"]):
                    debug("\t not available!")
                    continue

                resp = get_series(int(m["unit"]), int(m["t0"]), int(m["tf"]), int(m["dev"]), int(sig))[1]
                debug("\tgot", len(resp["series"]), " SmartData! Writing json to:", zip_path+"/data/"+m["measurement"]+".json")
                for mv in resp["series"]:
                    mv["value"] = decode_motion_vector(mv)
                with open(zip_path+"/data/"+m["measurement"]+"_"+str(sig)+".json", "w") as f:
                    f.write(json.dumps(resp, indent=4))

###########################################

if len(sys.argv) < 4:
    print("Expected use: ./iot_sdav_data_getter.py <commit_id : hash from git> <timestamp : microseconds since epoch> <download camera/lidar : 1/0>Missing argument 'commit_id' or 'timestamp' or , please add the hash of the commit followed by a space and then the timestamp of the commit as a parameter of this script!")
    sys.exit(1)

commit_id = sys.argv[1]
t0 = int(sys.argv[2])
download_cam_lidar = int(sys.argv[3])

zip_unit = 4 << 24 | 1

debug("Attempting to get zips from simulations ranging from", t0, "until", t0+1000000*60*MINUTES_IN_THE_FUTURE_TO_LOOK, ", using signature as:", SIGNATURE, ", if this is not the signature from the ego vehicle, please, edit this script to update this value!")
resp = get_series(zip_unit, t0-1000, t0+1000000*60*MINUTES_IN_THE_FUTURE_TO_LOOK, 0, SIGNATURE)[1]
debug("Done!")

size_resp = len(resp["series"])
if (size_resp == 0):
    debug(size_resp, " simulation executions found! Try other timestamps or increasing 'MINUTES_IN_THE_FUTURE_TO_LOOK'...")
    sys.exit(1)

debug(size_resp, ", simulation executions found found from ", t0, "until", t0+1000000*60*MINUTES_IN_THE_FUTURE_TO_LOOK)

#'''
# Example usage to download several commit_ids
original_zip_path = zip_path
for run in range(1, 221):
    debug("will now search for the one with commit_id =", commit_id)
    commit_id = str(run)
    zip_path = original_zip_path+commit_id
    if (not os.path.exists(zip_path)):
        os.mkdir(zip_path) 
    process_zips_smartdata(commit_id, resp, size_resp)

'''
# regular behavior, download commit_id passed as parameter
process_zips_smartdata(commit_id, resp, size_resp)
# '''