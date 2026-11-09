"""Read each stored local ZIP member and verify it against central-directory CRC.

Some WPS archives have offsets that Python's concatenation adjustment shifts by
one. Recover only exact named members with matching size and CRC; do not repair
or infer any presentation XML or asset payload.
"""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import copy, hashlib, json, struct, zlib

R=Path(__file__).resolve().parent
source=R/'before-additions.pptx'
data=source.read_bytes()
payloads=[]
report=[]
binary_repairs=[]
with ZipFile(source) as z:
 for info in z.infolist():
  candidates=[]
  for offset in (info.header_offset,info.header_offset+1,info.header_offset-1):
   if offset<0 or data[offset:offset+4]!=b'PK\x03\x04':continue
   _,version,flags,method,mt,md,crc,cs,us,nl,el=struct.unpack_from('<IHHHHHIIIHH',data,offset)
   name_bytes=data[offset+30:offset+30+nl]
   name=name_bytes.decode('utf-8' if flags & 2048 else 'cp437')
   if name!=info.filename:continue
   assert not flags & 1, 'encrypted member'
   start=offset+30+nl+el
   comp=data[start:start+info.compress_size]
   assert method==info.compress_type
   if method==0:plain=comp
   elif method==8:
    try:plain=zlib.decompress(comp,-15)
    except zlib.error as exc:
     # This input has one missing CR byte in a stored-deflate PNG signature.
     # Restore only that exact byte, and accept it only if the recorded full
     # member size and CRC match. The original backup is never changed.
     assert name=='ppt/media/image1.png' and comp.startswith(bytes.fromhex('000080ff7f89504e470a1a0a')), (name,str(exc))
     fixed=comp[:9]+b'\r'+comp[9:-1]
     plain=zlib.decompress(fixed,-15)
     binary_repairs.append({'part':name,'repair':'restore missing CR in PNG signature inside stored-deflate stream','originalByteOffset':start+9,'verifiedAgainstRecordedCrc':hex(info.CRC)})
   else:raise ValueError((name,method))
   assert len(plain)==info.file_size,(name,'size')
   assert zlib.crc32(plain)&0xffffffff==info.CRC,(name,'CRC')
   candidates.append((offset,plain))
  assert len(candidates)==1,(info.filename,len(candidates))
  offset,plain=candidates[0]
  payloads.append((copy.copy(info),plain))
  report.append({'part':info.filename,'offsetAdjustment':offset-info.header_offset,'uncompressedSha256':hashlib.sha256(plain).hexdigest()})
assert len(set(i.filename for i,b in payloads))==len(payloads)
out=R/'normalized.pptx'
with ZipFile(out,'w') as z:
 for info,plain in payloads:z.writestr(info,plain)
with ZipFile(out) as z:
 assert z.testzip() is None
 assert all(z.read(i.filename)==b for i,b in payloads)
(R/'normalization.json').write_text(json.dumps({'inputSha256':hashlib.sha256(data).hexdigest(),'outputSha256':hashlib.sha256(out.read_bytes()).hexdigest(),'allPayloadsMatchRecordedSizeAndCrc':True,'binaryRepairs':binary_repairs,'members':report},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'entries':len(report),'adjustedOffsets':sum(x['offsetAdjustment']!=0 for x in report),'payloadsVerified':True,'output':str(out)},ensure_ascii=False))
