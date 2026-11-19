"""
parse.py - NISAR Orbit Planner, step 1 of 2.

Read every daily orbit KMZ in the All_Pass folder (two levels above this script) and
extract each NISAR scene that intersects Thailand: its attributes (date, pass direction,
track, frame, radar mode, overlap area/percent), its full footprint and its footprint
clipped to Thailand. The Thailand boundary is read once from the first KMZ.

Output: nisar.json in the current working directory, used by build.py.

Usage:
    python parse.py
"""
# Standard-library modules: read KMZ archives, list files, write JSON, build paths and parse KML (XML).
import zipfile,glob,json,os,xml.etree.ElementTree as ET
# XML namespace prefix used by every KML element.
ns={'k':'http://www.opengis.net/kml/2.2'}
src=os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..'))  # folder containing the .kmz files
# Extract polygon coordinates from a KML element as [polygon][ring][[lon, lat], ...].
def coords(el):
    """Return the polygons under a KML element as [polygon][ring][[lon, lat], ...].

    Args:
        el: ElementTree element (a Placemark or a whole KML document).

    Returns:
        list: One list of rings per Polygon; coordinates rounded to 5 decimals.
    """
    # One entry per polygon.
    out=[]
    # Visit every Polygon element under this element.
    for poly in el.iter('{%s}Polygon'%ns['k']):
        # Rings of this polygon (outer boundary first, then holes).
        rings=[]
        # Visit every coordinates element of the polygon.
        for c in poly.iter('{%s}coordinates'%ns['k']):
            # Parse "lon,lat[,alt]" tuples and round lon/lat to 5 decimals (about 1 m).
            pts=[[round(float(a),5),round(float(b),5)] for a,b,*_ in (p.split(',') for p in c.text.split())]
            # Add the ring to the polygon.
            rings.append(pts)
        # Add the polygon to the output.
        out.append(rings)
    # Return all polygons.
    return out
# Collected scenes, and the Thailand boundary (read once).
scenes=[];border=None
# Read each daily orbit KMZ in date order.
for f in sorted(glob.glob(src+'/*.kmz')):
    # Open the KMZ (a zip archive of KML layers).
    z=zipfile.ZipFile(f)
    # Read the full-scene layer first, then the layer clipped to Thailand.
    for layer,key in [('Full scenes intersecting Thailand','full'),('Footprints clipped to Thailand','clip')]:
        # Parse the layer's KML document.
        r=ET.fromstring(z.read('layers/%s.kml'%layer))
        # Visit every scene placemark in the layer.
        for pm in r.iter('{%s}Placemark'%ns['k']):
            # Placemark attributes (ExtendedData SimpleData) as a name -> text dict.
            d={s.get('name'):s.text for s in pm.iter('{%s}SimpleData'%ns['k'])}
            # Polygon coordinates of the placemark.
            g=coords(pm)
            # Full-scene layer: create a new scene record.
            if key=='full':
                # Store the scene attributes, the full footprint and the source KMZ name.
                scenes.append(dict(id=pm.find('k:name',ns).text,date=d['acq_date'],dir=d['pass_dir'][0],track=int(d['track']),frame=int(d['frame']),scene=d['scene_name'],src=d['source_id'],mode=d.get('radar_mode') or '',km2=float(d.get('overlap_km2') or 0),pct=float(d.get('overlap_pct') or 0),full=g,file=os.path.basename(f)))
            # Clipped layer: attach the clipped footprint to its scene.
            else:
                # Scene name of this clipped placemark.
                nm=pm.find('k:name',ns).text
                # Find the matching scene from the same KMZ.
                for s in scenes:
                    # Store the clipped footprint on that scene.
                    if s['id']==nm and s['file']==os.path.basename(f): s['clip']=g
    # The Thailand boundary has not been read yet.
    if border is None:
        # Read the boundary layer once and keep its polygons.
        r=ET.fromstring(z.read('layers/Thailand boundary.kml')); border=coords(r)
# Write every scene and the boundary to nisar.json in compact form.
json.dump(dict(scenes=scenes,border=border),open('nisar.json','w'),separators=(',',':'))
# Print the scene count, scenes with a clipped footprint, and unique scene IDs.
print(len(scenes),sum('clip' in s for s in scenes),len(set(s['id'] for s in scenes)))
# Counter for the summary statistics below.
import collections
# Print the 15 most common radar modes.
print(collections.Counter(s['mode'] for s in scenes).most_common(15))
# Print every (pass direction, track) combination found.
print(sorted(set((s['dir'],s['track']) for s in scenes)))
