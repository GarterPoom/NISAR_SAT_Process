import json,math,os
from shapely.geometry import shape,mapping,Polygon,MultiPolygon,box
from shapely.ops import unary_union
from shapely import STRtree
BB=box(88,-2,116,29)
def r(g,nd=4):
    return json.loads(json.dumps(mapping(g)),parse_float=lambda x:round(float(x),nd))
def rnd(o,nd):
    if isinstance(o,float): return round(o,nd)
    if isinstance(o,(list,tuple)): return [rnd(x,nd) for x in o]
    if isinstance(o,dict): return {k:rnd(v,nd) for k,v in o.items()}
    return o
def km2(g):
    if g.is_empty: return 0
    c=g.centroid.y
    return g.area*111.32*111.32*math.cos(math.radians(c))
def fc(feats): return {"type":"FeatureCollection","features":feats}
# ---- basemap
cn=json.load(open('ne_10m_admin_0_countries.geojson'))
countries=[]
for f in cn['features']:
    g=shape(f['geometry'])
    if not g.intersects(BB): continue
    g=g.intersection(BB).simplify(0.01,preserve_topology=True)
    countries.append({"type":"Feature","properties":{"n":f['properties']['NAME_EN'] or f['properties']['NAME'],"a3":f['properties']['ADM0_A3']},"geometry":rnd(mapping(g),3)})
ad=json.load(open('ne_10m_admin_1_states_provinces.geojson'))
provs=[];pg=[]
for f in ad['features']:
    p=f['properties']
    if p['adm0_a3']!='THA': continue
    g=shape(f['geometry']).buffer(0)
    pg.append((p['name_en'] or p['name'],p['region'],g))
    provs.append({"type":"Feature","properties":{"n":p['name_en'] or p['name'],"th":p.get('name_local') or '',"rg":p['region'],"lat":p['latitude'],"lon":p['longitude']},"geometry":rnd(mapping(g.simplify(0.004,preserve_topology=True)),4)})
pp=json.load(open('ne_10m_populated_places_simple.geojson'))
places=[]
for f in pp['features']:
    p=f['properties'];x,y=p['longitude'],p['latitude']
    if not BB.contains(shape(f['geometry'])): continue
    if p['adm0_a3']!='THA' and p['scalerank']>3: continue
    places.append({"type":"Feature","properties":{"n":p['name'],"r":p['scalerank'],"cap":p['adm0cap'],"pop":p['pop_max']},"geometry":{"type":"Point","coordinates":[round(x,4),round(y,4)]}})
def lines(fn,filt,tol):
    out=[]
    for f in json.load(open(fn))['features']:
        if not f['geometry'] or not filt(f['properties']): continue
        g=shape(f['geometry'])
        if not g.intersects(BB): continue
        g=g.intersection(BB).simplify(tol)
        if g.is_empty: continue
        out.append({"type":"Feature","properties":{"n":f['properties'].get('name') or ''},"geometry":rnd(mapping(g),3)})
    return out
rivers=lines('ne_10m_rivers_lake_centerlines.geojson',lambda p:p.get('scalerank',99)<=9,0.005)
lakes=lines('ne_10m_lakes.geojson',lambda p:True,0.005)
roads=lines('ne_10m_roads.geojson',lambda p:p.get('type') in('Major Highway','Secondary Highway','Beltway') and p.get('continent')=='Asia',0.005)
open('basemap.js','w').write('window.BASEMAP='+json.dumps(dict(countries=fc(countries),provinces=fc(provs),places=fc(places),rivers=fc(rivers),lakes=fc(lakes),roads=fc(roads)),separators=(',',':'))+';')
print('countries',len(countries),'prov',len(provs),'places',len(places),'rivers',len(rivers),'lakes',len(lakes),'roads',len(roads))
# ---- scenes
d=json.load(open('nisar.json'))
def toshape(polys):
    ps=[Polygon(p[0],p[1:]) for p in polys]
    return unary_union([p.buffer(0) for p in ps]) if ps else Polygon()
tree=STRtree([g for _,_,g in pg])
border=unary_union([Polygon(p[0],p[1:]).buffer(0) for p in d['border']])
TH_KM2=km2(border)
out=[]
for s in d['scenes']:
    full=toshape(s['full']); clip=toshape(s.get('clip',[]))
    pl=[]
    if not clip.is_empty:
        for i in tree.query(clip):
            n,rg,g=pg[i]; a=km2(g.intersection(clip))
            if a>0.5: pl.append([n,round(a,1),round(100*a/km2(g),1)])
    pl.sort(key=lambda x:-x[1])
    c=full.centroid
    o=dict(s); o.pop('full');o.pop('clip',None);o.pop('file')
    o['prov']=pl; o['sceneKm2']=round(km2(full),0); o['c']=[round(c.x,3),round(c.y,3)]
    o['full']=rnd(mapping(full.simplify(0.0005)),4)
    o['clip']=rnd(mapping(clip.simplify(0.002,preserve_topology=True)),4) if not clip.is_empty else None
    out.append(o)
# per-date union coverage
import collections
by=collections.defaultdict(list)
for s,o in zip(d['scenes'],out):
    if o['clip']:
        g=shape(o['clip']).buffer(0); by[(o['date'],o['dir'])].append(g); by[(o['date'],'AD')].append(g)
cov={}
for k,v in by.items():
    cov[k[0]+'_'+k[1]]=round(100*km2(unary_union(v).intersection(border))/TH_KM2,1)
bnd=rnd(mapping(border.simplify(0.003,preserve_topology=True)),4)
open('data.js','w').write('window.NISAR='+json.dumps(dict(scenes=out,cov=cov,thKm2=round(TH_KM2),border=bnd),separators=(',',':'))+';')
print(len(out),TH_KM2,list(cov.items())[:4])
