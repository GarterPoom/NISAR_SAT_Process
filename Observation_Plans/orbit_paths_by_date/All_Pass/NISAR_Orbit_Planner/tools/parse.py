import zipfile,glob,json,os,xml.etree.ElementTree as ET
ns={'k':'http://www.opengis.net/kml/2.2'}
src=os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..'))  # folder containing the .kmz files
def coords(el):
    out=[]
    for poly in el.iter('{%s}Polygon'%ns['k']):
        rings=[]
        for c in poly.iter('{%s}coordinates'%ns['k']):
            pts=[[round(float(a),5),round(float(b),5)] for a,b,*_ in (p.split(',') for p in c.text.split())]
            rings.append(pts)
        out.append(rings)
    return out
scenes=[];border=None
for f in sorted(glob.glob(src+'/*.kmz')):
    z=zipfile.ZipFile(f)
    for layer,key in [('Full scenes intersecting Thailand','full'),('Footprints clipped to Thailand','clip')]:
        r=ET.fromstring(z.read('layers/%s.kml'%layer))
        for pm in r.iter('{%s}Placemark'%ns['k']):
            d={s.get('name'):s.text for s in pm.iter('{%s}SimpleData'%ns['k'])}
            g=coords(pm)
            if key=='full':
                scenes.append(dict(id=pm.find('k:name',ns).text,date=d['acq_date'],dir=d['pass_dir'][0],track=int(d['track']),frame=int(d['frame']),scene=d['scene_name'],src=d['source_id'],mode=d.get('radar_mode') or '',km2=float(d.get('overlap_km2') or 0),pct=float(d.get('overlap_pct') or 0),full=g,file=os.path.basename(f)))
            else:
                nm=pm.find('k:name',ns).text
                for s in scenes:
                    if s['id']==nm and s['file']==os.path.basename(f): s['clip']=g
    if border is None:
        r=ET.fromstring(z.read('layers/Thailand boundary.kml')); border=coords(r)
json.dump(dict(scenes=scenes,border=border),open('nisar.json','w'),separators=(',',':'))
print(len(scenes),sum('clip' in s for s in scenes),len(set(s['id'] for s in scenes)))
import collections
print(collections.Counter(s['mode'] for s in scenes).most_common(15))
print(sorted(set((s['dir'],s['track']) for s in scenes)))
