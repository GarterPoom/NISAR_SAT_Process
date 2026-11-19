"""
build.py - NISAR Orbit Planner, step 2 of 2.

Build the two JavaScript data files loaded by the planner's index.html:

* basemap.js - countries, Thai provinces, populated places, rivers, lakes and major
  roads clipped to the Thailand region, from Natural Earth 10 m GeoJSON files
  (ne_10m_*.geojson) that must be present in the current working directory.
* data.js - every scene from nisar.json (written by parse.py) with simplified
  footprints, the provinces each scene covers, and the percentage of Thailand
  covered per date and pass direction.

Requirements:
    pip install shapely

Usage:
    python build.py
"""
# Standard-library modules for JSON, maths and paths.
import json,math,os
# Shapely geometry types and GeoJSON conversion helpers.
from shapely.geometry import shape,mapping,Polygon,MultiPolygon,box
# unary_union merges many polygons into one geometry.
from shapely.ops import unary_union
# STRtree is a spatial index for fast "which provinces intersect?" queries.
from shapely import STRtree
# Bounding box (lon 88..116, lat -2..29) around Thailand and its neighbours.
BB=box(88,-2,116,29)
# Convert a geometry to GeoJSON with coordinates rounded to nd decimals.
def r(g,nd=4):
    """Return geometry ``g`` as a GeoJSON dict with coordinates rounded to ``nd`` decimals."""
    # Round floats while parsing the GeoJSON text back into Python objects.
    return json.loads(json.dumps(mapping(g)),parse_float=lambda x:round(float(x),nd))
# Recursively round every float in nested lists, tuples and dicts.
def rnd(o,nd):
    """Recursively round every float inside ``o`` (lists, tuples, dicts) to ``nd`` decimals."""
    # Round a float.
    if isinstance(o,float): return round(o,nd)
    # Round each item of a list or tuple.
    if isinstance(o,(list,tuple)): return [rnd(x,nd) for x in o]
    # Round each value of a dict.
    if isinstance(o,dict): return {k:rnd(v,nd) for k,v in o.items()}
    # Leave any other value unchanged.
    return o
# Approximate area of a lon/lat geometry in square kilometres.
def km2(g):
    """Approximate area of a lon/lat geometry in km2 (scaled by cos of the centroid latitude)."""
    # An empty geometry has no area.
    if g.is_empty: return 0
    # Latitude of the centroid, used for the longitude scale factor.
    c=g.centroid.y
    # Degrees squared x (111.32 km)^2 x cos(latitude) gives km2.
    return g.area*111.32*111.32*math.cos(math.radians(c))
# Wrap a list of features in a GeoJSON FeatureCollection.
def fc(feats):
    """Wrap a list of GeoJSON features in a FeatureCollection."""
    # Return the GeoJSON FeatureCollection dict.
    return {"type":"FeatureCollection","features":feats}
# ---- basemap
# Load the Natural Earth country polygons.
cn=json.load(open('ne_10m_admin_0_countries.geojson'))
# Simplified country features for the basemap.
countries=[]
# Process each country.
for f in cn['features']:
    # Shapely geometry of the country.
    g=shape(f['geometry'])
    # Skip countries outside the bounding box.
    if not g.intersects(BB): continue
    # Clip to the bounding box and simplify for a small file.
    g=g.intersection(BB).simplify(0.01,preserve_topology=True)
    # Store the English name, ISO code and rounded geometry.
    countries.append({"type":"Feature","properties":{"n":f['properties']['NAME_EN'] or f['properties']['NAME'],"a3":f['properties']['ADM0_A3']},"geometry":rnd(mapping(g),3)})
# Load the Natural Earth first-level admin units (provinces).
ad=json.load(open('ne_10m_admin_1_states_provinces.geojson'))
# Province features for the basemap, and (name, region, geometry) tuples for overlap statistics.
provs=[];pg=[]
# Process each admin unit.
for f in ad['features']:
    # Properties of this admin unit.
    p=f['properties']
    # Keep only Thai provinces.
    if p['adm0_a3']!='THA': continue
    # Shapely geometry, repaired with buffer(0).
    g=shape(f['geometry']).buffer(0)
    # Keep the full-detail geometry for the area calculations.
    pg.append((p['name_en'] or p['name'],p['region'],g))
    # Store names, region, label position and a simplified geometry for display.
    provs.append({"type":"Feature","properties":{"n":p['name_en'] or p['name'],"th":p.get('name_local') or '',"rg":p['region'],"lat":p['latitude'],"lon":p['longitude']},"geometry":rnd(mapping(g.simplify(0.004,preserve_topology=True)),4)})
# Load the Natural Earth populated places.
pp=json.load(open('ne_10m_populated_places_simple.geojson'))
# Place point features for the basemap.
places=[]
# Process each place.
for f in pp['features']:
    # Place properties and coordinates.
    p=f['properties'];x,y=p['longitude'],p['latitude']
    # Skip places outside the bounding box.
    if not BB.contains(shape(f['geometry'])): continue
    # Outside Thailand keep only major places (scale rank 3 or lower).
    if p['adm0_a3']!='THA' and p['scalerank']>3: continue
    # Store name, rank, capital flag, population and rounded point.
    places.append({"type":"Feature","properties":{"n":p['name'],"r":p['scalerank'],"cap":p['adm0cap'],"pop":p['pop_max']},"geometry":{"type":"Point","coordinates":[round(x,4),round(y,4)]}})
# Load a line/polygon layer, filter it, clip it to the bounding box and simplify it.
def lines(fn,filt,tol):
    """Load a GeoJSON layer, keep features accepted by ``filt``, clip them to BB and simplify.

    Args:
        fn: GeoJSON filename.
        filt: Function taking a feature's properties and returning True to keep it.
        tol: Simplification tolerance in degrees.

    Returns:
        list: GeoJSON features with a name property and rounded geometry.
    """
    # Features to return.
    out=[]
    # Process each feature of the GeoJSON file.
    for f in json.load(open(fn))['features']:
        # Skip features without geometry or rejected by the filter.
        if not f['geometry'] or not filt(f['properties']): continue
        # Shapely geometry of the feature.
        g=shape(f['geometry'])
        # Skip features outside the bounding box.
        if not g.intersects(BB): continue
        # Clip to the bounding box and simplify with the given tolerance.
        g=g.intersection(BB).simplify(tol)
        # Skip features that vanish after clipping.
        if g.is_empty: continue
        # Store the name and rounded geometry.
        out.append({"type":"Feature","properties":{"n":f['properties'].get('name') or ''},"geometry":rnd(mapping(g),3)})
    # Return the processed features.
    return out
# Major rivers and lake centre lines.
rivers=lines('ne_10m_rivers_lake_centerlines.geojson',lambda p:p.get('scalerank',99)<=9,0.005)
# All lakes.
lakes=lines('ne_10m_lakes.geojson',lambda p:True,0.005)
# Major and secondary highways and beltways in Asia.
roads=lines('ne_10m_roads.geojson',lambda p:p.get('type') in('Major Highway','Secondary Highway','Beltway') and p.get('continent')=='Asia',0.005)
# Write all basemap layers to basemap.js as a JavaScript global.
open('basemap.js','w').write('window.BASEMAP='+json.dumps(dict(countries=fc(countries),provinces=fc(provs),places=fc(places),rivers=fc(rivers),lakes=fc(lakes),roads=fc(roads)),separators=(',',':'))+';')
# Print how many features each layer has.
print('countries',len(countries),'prov',len(provs),'places',len(places),'rivers',len(rivers),'lakes',len(lakes),'roads',len(roads))
# ---- scenes
# Load the scenes and boundary written by parse.py.
d=json.load(open('nisar.json'))
# Convert [polygon][ring] coordinate lists into one shapely geometry.
def toshape(polys):
    """Merge [polygon][ring] coordinate lists into one shapely geometry (empty if none)."""
    # Build a polygon per entry: first ring is the outer boundary, the rest are holes.
    ps=[Polygon(p[0],p[1:]) for p in polys]
    # Merge the repaired polygons, or return an empty polygon.
    return unary_union([p.buffer(0) for p in ps]) if ps else Polygon()
# Spatial index over the province geometries.
tree=STRtree([g for _,_,g in pg])
# Thailand boundary as one merged geometry.
border=unary_union([Polygon(p[0],p[1:]).buffer(0) for p in d['border']])
# Area of Thailand in km2.
TH_KM2=km2(border)
# Processed scene records.
out=[]
# Process each scene.
for s in d['scenes']:
    # Full and clipped footprints as shapely geometries.
    full=toshape(s['full']); clip=toshape(s.get('clip',[]))
    # Provinces covered: [name, km2 covered, percent of province].
    pl=[]
    # Only scenes with a clipped footprint can cover provinces.
    if not clip.is_empty:
        # Candidate provinces whose bounding boxes meet the footprint.
        for i in tree.query(clip):
            # Province name, region, geometry and the km2 covered by the scene.
            n,rg,g=pg[i]; a=km2(g.intersection(clip))
            # Ignore slivers below 0.5 km2.
            if a>0.5: pl.append([n,round(a,1),round(100*a/km2(g),1)])
    # Sort provinces by covered area, largest first.
    pl.sort(key=lambda x:-x[1])
    # Centroid of the full footprint.
    c=full.centroid
    # Copy the scene and drop the raw polygons and source filename.
    o=dict(s); o.pop('full');o.pop('clip',None);o.pop('file')
    # Add province coverage, scene area and centroid.
    o['prov']=pl; o['sceneKm2']=round(km2(full),0); o['c']=[round(c.x,3),round(c.y,3)]
    # Simplified full footprint as GeoJSON.
    o['full']=rnd(mapping(full.simplify(0.0005)),4)
    # Simplified clipped footprint, or None.
    o['clip']=rnd(mapping(clip.simplify(0.002,preserve_topology=True)),4) if not clip.is_empty else None
    # Keep the processed scene.
    out.append(o)
# per-date union coverage
# defaultdict groups footprints by date and pass direction.
import collections
# (date, direction) -> list of clipped footprints.
by=collections.defaultdict(list)
# Walk the raw and processed scenes together.
for s,o in zip(d['scenes'],out):
    # Only scenes that cover Thailand.
    if o['clip']:
        # Add the footprint to its (date, direction) group and to the (date, both directions) group.
        g=shape(o['clip']).buffer(0); by[(o['date'],o['dir'])].append(g); by[(o['date'],'AD')].append(g)
# Coverage percent of Thailand per "date_direction" key.
cov={}
# Process each group.
for k,v in by.items():
    # Percentage of Thailand covered by the union of the group's footprints.
    cov[k[0]+'_'+k[1]]=round(100*km2(unary_union(v).intersection(border))/TH_KM2,1)
# Simplified Thailand boundary for display.
bnd=rnd(mapping(border.simplify(0.003,preserve_topology=True)),4)
# Write scenes, coverage, Thailand area and boundary to data.js as a JavaScript global.
open('data.js','w').write('window.NISAR='+json.dumps(dict(scenes=out,cov=cov,thKm2=round(TH_KM2),border=bnd),separators=(',',':'))+';')
# Print the scene count, Thailand area and the first coverage values.
print(len(out),TH_KM2,list(cov.items())[:4])
