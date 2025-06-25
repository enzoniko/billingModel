import requests
import json
import sys

NOT_SEND = True

class Unit:
    def __init__(self, unit):
        self.unit = unit
        self.si = (unit >> 31) & 0x01         # Bit 31
        if self.si == 1:
            # SI unit encoding (custom bitfield)
            self.length = unit & 0xFF             # Bits 0-7: length (if needed)
            self.mod = (unit >> 8) & 0xFF         # Bits 8-15: modifier
            self.sr = (unit >> 24) & 0x07         # Bits 24-26: steradian exponent (3 bits)
            self.rad = (unit >> 21) & 0x07        # Bits 21-23: radian exponent (3 bits)
            self.m = (unit >> 18) & 0x07          # Bits 18-20: meter exponent (3 bits)
            self.kg = (unit >> 15) & 0x07         # Bits 15-17: kilogram exponent (3 bits)
            self.s = (unit >> 12) & 0x07          # Bits 12-14: second exponent (3 bits)
            self.a = (unit >> 9) & 0x07           # Bits 9-11: ampere exponent (3 bits)
            self.k = (unit >> 6) & 0x07           # Bits 6-8: kelvin exponent (3 bits)
            self.mol = (unit >> 3) & 0x07         # Bits 3-5: mole exponent (3 bits)
            self.cd = unit & 0x07                 # Bits 0-2: candela exponent (3 bits)
            self.length = 0           # Bits 0-15
            self.subtype = 0    # Bits 16-23
            self.type = 0       # Bits 24-28 (5 bits)
            self.multi = 0      # Bits 29-30 (2 bits)
        else:
            self.length = unit & 0xFFFF           # Bits 0-15
            self.subtype = (unit >> 16) & 0xFF    # Bits 16-23
            self.type = (unit >> 24) & 0x1F       # Bits 24-28 (5 bits)
            self.multi = (unit >> 29) & 0x03      # Bits 29-30 (2 bits)

    @staticmethod
    def build_digital_unit(length=0, subtype=0, type=0, multi=0):
        """
        Build a 32-bit integer representing a non-SI (digital) unit.
        """
        unit = 0
        unit |= (multi & 0x03) << 29      # multiplier (2 bits)
        unit |= (type & 0x1F) << 24       # type (5 bits)
        unit |= (subtype & 0xFF) << 16    # subtype (8 bits)
        unit |= (length & 0xFFFF)         # length (16 bits)
        # SI bit is 0 by default
        return unit

    @staticmethod
    def build_si_unit(length=0, mod=0, sr=0, rad=0, m=0, kg=0, s=0, a=0, k=0, mol=0, cd=0):
        """
        Build a 32-bit integer representing an SI unit using the custom bitfield encoding.
        """
        unit = 0
        unit |= (1 << 31)                # SI bit
        unit |= (sr & 0x07) << 24        # steradian exponent (3 bits)
        unit |= (rad & 0x07) << 21       # radian exponent (3 bits)
        unit |= (m & 0x07) << 18         # meter exponent (3 bits)
        unit |= (kg & 0x07) << 15        # kilogram exponent (3 bits)
        unit |= (s & 0x07) << 12         # second exponent (3 bits)
        unit |= (a & 0x07) << 9          # ampere exponent (3 bits)
        unit |= (k & 0x07) << 6          # kelvin exponent (3 bits)
        unit |= (mol & 0x07) << 3        # mole exponent (3 bits)
        unit |= (cd & 0x07)              # candela exponent (3 bits)
        unit |= (mod & 0xFF) << 8        # modifier (8 bits)
        unit |= (length & 0xFF)          # length (8 bits, if needed)
        return unit

    def __repr__(self):
        if (self.si == 1):
            return (f"Unit(unit={self.unit}, length={self.length}, mod={self.mod}, "
                    f"sr={self.sr}, rad={self.rad}, m={self.m}, kg={self.kg}, "
                    f"s={self.s}, a={self.a}, k={self.k}, mol={self.mol}, cd={self.cd})")
        else:
            return (f"Unit(unit={self.unit}, length={self.length}, subtype={self.subtype}, "
                    f"type={self.type}, multi={self.multi}, si={self.si})")

class DB_Record:
    def __init__(self, version=0, unit=0, value=0, error=0, confidence=0, x=0, y=0, z=0, t=0, signature=0, dev=0):
        self.version = version
        self.unit = unit
        self.value = value
        self.error = error
        self.confidence = confidence
        self.x = x
        self.y = y
        self.z = z
        self.t = t
        self.signature = signature
        self.dev = dev

    def to_dict(self):
        return {
            'version': self.version,
            'unit': self.unit,
            'value': self.value,
            'error': self.error,
            'confidence': self.confidence,
            'x': self.x,
            'y': self.y,
            'z': self.z,
            't': self.t,
            'signature': self.signature,
            'dev': self.dev
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            version=data.get('version'),
            unit=data.get('unit'),
            value=data.get('value'),
            error=data.get('error'),
            confidence=data.get('confidence'),
            x=data.get('x'),
            y=data.get('y'),
            z=data.get('z'),
            t=data.get('t'),
            signature=data.get('signature'),
            dev=data.get('dev')
        )
    
class DB_Series:
    def __init__(self, version, unit, t0, t1, workflow, dev, signature):
        self.version = version
        self.unit = unit
        self.t0 = t0
        self.t1 = t1
        self.workflow = workflow
        self.dev = dev
        self.signature = signature

    def to_dict(self):
        return {
            'version': self.version,
            'unit': self.unit,
            't0': self.t0,
            't1': self.t1,
            'workflow': self.workflow,
            'dev': self.dev,
            'signature': self.signature
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            version=data.get('version'),
            unit=data.get('unit'),
            t0=data.get('t0'),
            t1=data.get('t1'),
            workflow=data.get('workflow'),
            dev=data.get('dev'),
            signature=data.get('signature')
        )

class SmartData_IoT_Client:
    MOBILE = "1.2"
    STATIC = "1.1"
    def __init__(self, host='deviot.setic.ufsc.br', my_certificate=None, debug_mode=False, log_path='iot_debug.log', buffer_max_len=100, retries=5):
        self.HOST = host
        self.CREATE_URL = f'https://{self.HOST}/api/create.php'
        self.PUT_URL = f'https://{self.HOST}/api/put.php'
        self.GET_URL = f'https://{self.HOST}/api/get.php'
        self.recreate_session(my_certificate)
        self.buffer = {}
        self.DEBUG = debug_mode
        self.log_path = log_path
        self.BUFFER_MAX_LEN = buffer_max_len
        self.retries = retries

    @staticmethod
    def create_series_json(first_data : DB_Record, last_data : DB_Record):
        series = {'series': DB_Series(version=SmartData_IoT_Client.MOBILE, unit=first_data.unit, t0=first_data.t, t1=last_data.t, workflow=0, dev=first_data.dev, signature=first_data.signature).to_dict() }
        return series
    
    def recreate_session(self, my_certificate):
        self.session = requests.Session()
        self.session.headers = {'Content-type': 'application/json'}
        self.session.cert = my_certificate

    def debug(self, *args):
        if self.DEBUG:
            print(*args)
        output = ' '.join(map(str, args))
        output = output + "\n"
        with open(self.log_path, "a") as log_file:
            log_file.write(output)

    def send(self, url, content):
        response = self.session.post(url, json.dumps(content))
        rc = response.status_code
        return rc

    def __send_series_and_sd(self, series, SDs):
        if NOT_SEND:
            self.debug("NOT_SEND is set to True, skipping send operation")
            # For debugging purposes, we return a default response code 
            return 1, 1 # default response codes for debugging
        retries = self.retries
        response1 = 400
        while(response1 > 300 and retries > 0):
            response1 = self.send(self.CREATE_URL, series)
            retries -= 1
        retries = self.retries
        response2 = 400
        sds = {'smartdata': [sd.to_dict() for sd in SDs]}
        while(response2 > 300 and retries > 0):
            response2 = self.send(self.PUT_URL, sds)
            retries -= 1
        return response1, response2
    
    def send_with_retries(self, SDs):
        series = SmartData_IoT_Client.create_series_json(SDs[0], SDs[-1])
        response1, response2 = self.__send_series_and_sd(series, SDs)
        
        if response1 > 300 or response2 > 300:
            self.debug("ERROR", f"Fail to send dev: {SDs[0].dev} buffer after retries ({self.retries}) (series_resp_code = {response1}, data_resp_code={response2})")
            self.debug("recreating session...")
            self.recreate_session(self.session.cert)
            self.debug("Done! Let's try again...")
            response1, response2 = self.__send_series_and_sd(series, SDs)
            if response1 > 300 or response2 > 300:
                self.debug("ERROR", f"Fail to send dev: {SDs[0].dev} buffer after retries ({self.retries*2}) (series_resp_code = {response1}, data_resp_code={response2})")
            else:
                self.debug(f"Dev {SDs[0].dev} of size {len(SDs)} sent successfully (series_resp_code = {response1}, data_resp_code={response2})")
        else:
            self.debug(f"Dev {SDs[0].dev} of size {len(SDs)} sent successfully (series_resp_code = {response1}, data_resp_code={response2})")

    def send_to_iot(self, SD : DB_Record, FORCE=False):
        dev = SD.dev
        if dev in self.buffer:
            self.buffer[dev].append(SD)
        else:
            self.buffer[dev] = [SD]
        if len(self.buffer[dev]) >= self.BUFFER_MAX_LEN:
            data = self.buffer[dev]
            self.send_with_retries(self.buffer[dev])
            del self.buffer[dev]

        if FORCE:
            for dev in list(self.buffer.keys()):
                data = self.buffer[dev]
                if len(data) == 0:
                    continue
                self.send_with_retries(data)
            self.buffer.clear()

