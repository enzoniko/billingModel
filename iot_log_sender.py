# -*- coding: utf-8 -*-

import json
import re
import base64
import pyproj
import time
import sys
import struct
from smartdata import SmartData_IoT_Client, DB_Record, DB_Series, Unit
from sniffer_cleaner import remove_content_after_pattern

''' Define constants '''
LOG_PATH="./log.txt"
#clean log 
with open(LOG_PATH, "w") as log_file:
    log_file.write("")
DEBUG = True, # Set to True for debugging, False for production -- enables printing debug messages
# Devices that use relative motion vectors
RELATIVE_MV_DEVICES = [20, 21, 22, 23] # CAMERA, LIDAR, RADAR, and FUSER
ETSI_MV_DEVICES = [26,27,28] # ETSI motion vector devices
EGO_CLASS = 12    # Class for ego motion vectors (unit.subtype)

wgs84_to_ecef = pyproj.Transformer.from_crs(
    {"proj":'latlong', "ellps":'WGS84', "datum":'WGS84'},
    {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
    )

ecef_to_wgs84 = pyproj.Transformer.from_crs(
    {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
    {"proj":'latlong', "ellps":'WGS84', "datum":'WGS84'},
    )

''' Funtions'''

# This function will print debug messages to the console if DEBUG is True
def debug(*args):
    if (DEBUG):
        print(*args)
    
    output = ' '.join(map(str, args))
    output = output + "\n"
    with open(LOG_PATH, "a") as log_file:
        log_file.write(output)

def convert_to_valid_json(text):
    # Replace single quotes with double quotes and ensure keys are double-quoted
    text = text.replace("'", '"')
    text = re.sub(r'(\w+)=', r'"\1":', text)
    return text

def get_entries_from_dirty_sniffer_log(file_path, save_path):
    """
    Reads the sniffer log file and extracts entries based on the defined regex.
    Returns a list of DB_Record objects.
    """
    # Regex to match each entry and extract values
    ENTRY_REGEX = re.compile(
        r"u=\{[^}]+\}=>"
        r"(?P<u>\d+),"
        r"d=(?P<d>\d+),"
        r"t=(?P<t>\d+),"
        r"sig=(?P<sig>\d+)\)?"
        r"=\{(?P<v>[^\[\{]*?)(\[(?P<list_v>[^\]]*)\])?\}"
    )
    clean_sniffer = remove_content_after_pattern(file_path, save_path)
    start_string = "Log Start:\n"
    log_start = clean_sniffer.index(start_string)

    clean_sniffer = clean_sniffer[log_start+len(start_string):]
    entries = {}
    for match in ENTRY_REGEX.finditer(clean_sniffer.replace(",\n\t]","\n\t]")):
        if match.group("list_v"):
            list_v = match.group("list_v").strip()
            # Convert the string list to an actual list of dictionaries
            list_v = convert_to_valid_json(f"[{list_v}]")
            list_v = json.loads(list_v)
            entry = DB_Record(unit=Unit(int(match.group("u"))), dev=int(match.group("d")), t=int(match.group("t")), signature=int(match.group("sig")), value=list_v)
        else:
            v = float(match.group("v")) if match.group("v").strip() else None
            entry = DB_Record(unit=Unit(int(match.group("u"))), dev=int(match.group("d")), t=int(match.group("t")), signature=int(match.group("sig")), value=v)
        if (not entry.signature in entries):
            entries[entry.signature] = {}
            debug("new sig=", entry.signature)
        if (not entry.dev in entries[entry.signature]):
            entries[entry.signature][entry.dev] = []
            debug("new dev=", entry.dev)
        entries[entry.signature][entry.dev].append(entry)
    return entries

def filter_ego_motion_vectors(entries):
    return entries[16]
    _e = []
    for e in entries:
        unit = e.unit
        if (unit.si == 1):
            continue
        else:
            if (unit.subtype == EGO_CLASS):
                _e.append(e)
    return _e

def mv_to_bytes(mv):
    byte_array = b''
    byte_array += struct.pack('1f', mv['speed'])
    byte_array += struct.pack('1f', mv['heading'])
    byte_array += struct.pack('1f', mv['yawr'])
    byte_array += struct.pack('1f', mv['accel'])
    byte_array += struct.pack('Q', mv['id'])
    return byte_array

def relative_mv_to_bytes(mv, ego_speed, ego_heading, ego_yawr, ego_accel):
    byte_array = b''
    byte_array += struct.pack('1f', mv['speed'] + ego_speed)
    byte_array += struct.pack('1f', mv['heading'] + ego_heading)
    byte_array += struct.pack('1f', mv['yawr'] + ego_yawr)
    byte_array += struct.pack('1f', mv['accel'] + ego_accel)
    byte_array += struct.pack('Q', mv['id'])
    return byte_array

def convert_bytes_to_json_blob(bytes):
    binstr = base64.b64encode(bytes)
    aux = str(binstr)
    return aux[2 : len(aux)-1]

def get_ecef_position(ego_mv):
    lon = int(ego_mv.value[0]['lon']) / 100000000.
    lat = int(ego_mv.value[0]['lat']) / 100000000.
    alt = int(ego_mv.value[0]['alt']) / 1000.
    x, y, z = wgs84_to_ecef.transform(lat, lon, alt)
    return x, y, z

def parse_egos_ECEF_position(egos, is_time_step_fix, ts_step, signature):
    """
    Parses DB_Record of mv and update the position in ECEF coordinates.
    """
    t0 = egos[0].t
    for index, mv in enumerate(egos):
        x, y, z = get_ecef_position(mv)
        mv.x = x
        mv.y = y
        mv.z = z
        if (is_time_step_fix):
            mv.t = t0 + index * ts_step
        mv.signature = signature
    return egos

def find_latest_before(records, current_t):
    """
    Given a list of DB_Record elements, finds the one with the largest t less than current_t.
    Returns a tuple (element, index) where element is the found record and index is its position in the records list.
    Returns (records[0], 0) if no such record exists.
    """
    filtered = [(r, i) for i, r in enumerate(records) if r.t < current_t]
    if not filtered:
        return records[0], 0
    return max(filtered, key=lambda x: x[0].t)

def process_and_send_si_when_ego_is_available(sds, egos, IoT_Client, is_time_step_fix, ts_step, signature):
    if (sds[0].unit.si != 1):
        debug("process_and_send_si_when_ego_is_available for non-si unit=", sds[0].unit)
    else:
        t0 = sds[0].t
        _, ego_index = find_latest_before(egos, t0)        
        for index, e in enumerate(sds):
            if is_time_step_fix:
                e.t = t0 + index * ts_step
            if (e.t > egos[ego_index].t and ego_index < len(egos) - 1 and e.t < egos[ego_index+1].t):
                ego_index += 1
            e.x = egos[ego_index].x
            e.y = egos[ego_index].y
            e.z = egos[ego_index].z
            e.signature = signature
        IoT_Client.send_with_retries(sds)

def process_and_send_motion_vector_when_ego_is_available(sds, dev, egos, IoT_Client, is_time_step_fix, ts_step, signature):
    if (sds[0].unit.si == 1 or sds[0].unit.multi != 1):
        debug("process_and_send_motion_vector_list_when_ego_is_available for non-multi unit=", sds[0].unit,", dev=", dev)
    else:
        debug("process_and_send_motion_vector_list_when_ego_is_available for multi unit=", sds[0].unit,", dev=", dev, "with", len(sds), "entries")
        records = []
        t0 = sds[0].t
        _, ego_index = find_latest_before(egos, t0)
        for index, e in enumerate(sds):
            if is_time_step_fix:
                e.t = t0 + index * ts_step
            if (e.t > egos[ego_index].t and ego_index < len(egos) - 1 and e.t < egos[ego_index+1].t):
                ego_index += 1
            # debug("entry[", index, "] with", len(e.value), "motion vectors")

            for data in e.value: # for each motion vector in the list
                mv = data # get dict of data
                u = Unit.build_digital_unit(1, mv['class'], 0, 1) # # fix unit length as it is sent as a single motion vector
                loc_lon = int(mv['lon'])/100000000.
                loc_lat = int(mv['lat'])/100000000.
                loc_alt = int(mv['alt'])/1000.
                _x, _y, _z = wgs84_to_ecef.transform(loc_lon,loc_lat,loc_alt,radians = True)
                if (dev not in RELATIVE_MV_DEVICES):
                    binstr = mv_to_bytes(mv)
                else:
                    binstr = relative_mv_to_bytes(mv, float(egos[ego_index].value[0]["speed"]), 
                                                  float (egos[ego_index].value[0]["heading"]), 
                                                  float(egos[ego_index].value[0]["yawr"]), 
                                                  float(egos[ego_index].value[0]["accel"]))
                    _x=int(_x) + egos[ego_index].x,
                    _y=int(_y) + egos[ego_index].y,
                    _z=int(_z) + egos[ego_index].z,
                v = convert_bytes_to_json_blob(binstr)
                records.append(DB_Record(
                    version=SmartData_IoT_Client.MOBILE,
                    unit=u,
                    value=v,
                    x=int(_x),
                    y=int(_y),
                    z=int(_z),
                    t=e.t,
                    signature=signature,
                    dev=dev
                ))
        if (len(records) == 0):
            debug("No motion vectors found for device", dev)
        else:
            IoT_Client.send_with_retries(records)

def process_and_send_ego_motion_vectors(mvs, IoT_Client):
    for e in mvs:
        mv = e.value[0]
        binstr = mv_to_bytes(mv)
        e.value = convert_bytes_to_json_blob(binstr)
        e.unit = Unit.build_digital_unit(1, mv['class'], 0, 1)
    if (len(mvs) > 0):
        IoT_Client.send_with_retries(mvs)

def process_and_send_ETSI_motion_vectors(mvs, IoT_Client, is_time_step_fix, ts_step):
    t0 = mvs[0].t
    for index, e in enumerate(mvs):
        e.unit = Unit.build_digital_unit(1, mv['class'], 0, 1) # fix unit length as it is sent as a single motion vector
        mv = e.value[0]
        e.signature = mv["id"]
        binstr = mv_to_bytes(mv)
        e.value = convert_bytes_to_json_blob(binstr)
        x, y, z = get_ecef_position(mv)
        e.x = x
        e.y = y
        e.z = z
        if (is_time_step_fix):
            e.t = t0 + index * ts_step
    if (len(mvs) > 0):
        IoT_Client.send_with_retries(mvs)

def parse_log_and_send(is_time_step_fix = True, # Set to True if the time step is fixed, False otherwise
                       ts_step = 100000,  # Time step is 100 milliseconds
                        # Paths for logs
                        smartdata_log_path = '/home/lisha/sdav_integration/smartdata/logs/',
                        sniffer_log_path = '/home/lisha/sdav_integration/smartdata/logs/sniffer.log',
                        # Certificate paths + network host
                        my_certificate = ('/home/lisha/sdav_integration/sdav_cert/sdav.pem', '/home/lisha/sdav_integration/sdav_cert/sdav.key'),
                        host = 'deviot.setic.ufsc.br',
                        buffer_max_len = 1000, # Maximum length of the buffer for SmartData_IoT_Client
                        signature = 400   # default vehicle signature
                        ):
    
    #try:
    entries = get_entries_from_dirty_sniffer_log(sniffer_log_path, smartdata_log_path)
    #except Exception as e:
    #    debug(f"An error occurred: {e}")
    #    sys.exit(1)

    veh_sig_list = [signature]

    IoT_Client = SmartData_IoT_Client(
        host=host,
        my_certificate=my_certificate,
        debug_mode=DEBUG,
        log_path=LOG_PATH,
        buffer_max_len=buffer_max_len
    )

    for sig in entries:
        debug ("sig=", sig, "len=", len(entries[sig]))

        egos = filter_ego_motion_vectors(entries[sig])
        egos = parse_egos_ECEF_position(egos, is_time_step_fix, ts_step, signature)

        if len(egos) == 0:
            debug("No ego motion vectors found for signature", sig)
            continue

        for d in entries[sig]:
            if d == 16:
                debug("Skipping device 16, we will send it later")
                continue
            if (d in ETSI_MV_DEVICES):
                debug("Processing ETSI motion vector device", d, "with unit=", entries[sig][d][0].unit, "and signature=", entries[sig][d][0].value[0]["id"])
                veh_sig_list.append(entries[sig][d][0].value[0]["id"])
                process_and_send_ETSI_motion_vectors(entries[sig][d], IoT_Client, is_time_step_fix, ts_step)
            if entries[sig][d][0].unit.si == 1:
                debug("Processing si device = ", d, " with unit=", entries[sig][d][0].unit)
                process_and_send_si_when_ego_is_available(entries[sig][d], egos, IoT_Client, is_time_step_fix, ts_step, signature)
            else:
                debug("Processing non-si device = ", d, " with unit=", entries[sig][d][0].unit)
                process_and_send_motion_vector_when_ego_is_available(entries[sig][d], d, egos, IoT_Client, is_time_step_fix, ts_step, signature)
            
        debug("Sending ego motion vectors for signature", sig)
        process_and_send_ego_motion_vectors(egos, IoT_Client)
        debug("Finished processing signature", sig)

if __name__ == "__main__":
    # Example usage
    missing = [54, 55]
    for i in range(1, 221):
        if (i in missing):
            continue
        parse_log_and_send(
            is_time_step_fix=True,  # Set to True if the time step is fixed, False otherwise
            ts_step=100000,  # Time step is 100 milliseconds
            smartdata_log_path='data/',
            sniffer_log_path='data/sniffer_run_'+str(i)+'.log',
            my_certificate=('sdav_cert/sdav.pem', '/sdav_cert/sdav.key'),
            host='deviot.setic.ufsc.br',
            buffer_max_len=1000,  # Maximum length of the buffer for SmartData_IoT_Client
            signature=400  # Default vehicle signature
        )

