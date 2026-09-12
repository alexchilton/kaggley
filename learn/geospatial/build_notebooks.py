"""Fill the Geospatial Analysis exercise notebooks.

Answers come from learntools/geospatial/ex{1..5}.py.

Almost every question in this course is a `CodingProblem` whose check only
confirms that the variable exists and has the right type, or that a map object
was built. A few are stricter: ex2 q_7 compares a float, and ex5 q_6 asserts
the proposed hospital sites actually bring the out-of-range percentage below
10%, which is a real computation over the data rather than a fixed value.

Note that in this course the answer cell is often NOT the cell that calls
check() -- the map-building cells come first and the check sits alone
afterwards. Auditing only check cells would miss them, which is why
kaggle/_tools/audit.py also reports any `____` left anywhere.

    python kaggle/geospatial/build_notebooks.py
"""

import json
import re
from pathlib import Path

HERE = Path(__file__).parent

CELLS = {
"exercise-your-first-map.ipynb": {
 4: '''loans_filepath = "../input/geospatial-learn-course-data/kiva_loans/kiva_loans/kiva_loans.shp"

# Your code here: Load the data
world_loans = gpd.read_file(loans_filepath)

# Check your answer
q_1.check()

# Uncomment to view the first five rows of the data
#world_loans.head()
''',
 9: '''# Your code here
ax = world.plot(figsize=(20,20), color='whitesmoke', linestyle=':', edgecolor='black')
world_loans.plot(ax=ax, markersize=2)

# Uncomment to see a hint
#q_2.hint()
''',
 12: '''# Your code here
PHL_loans = world_loans.loc[world_loans.country=="Philippines"].copy()

# Check your answer
q_3.check()
''',
 17: '''# Your code here
ax = PHL.plot(figsize=(12,12), color='whitesmoke', linestyle=':', edgecolor='lightgray')
PHL_loans.plot(ax=ax, markersize=2)

# Uncomment to see a hint
#q_4.a.hint()
''',
},
"exercise-coordinate-reference-systems.ipynb": {
 6: '''# Your code here: Create the GeoDataFrame
birds = gpd.GeoDataFrame(birds_df, geometry=gpd.points_from_xy(birds_df["location-long"], birds_df["location-lat"]))

# Your code here: Set the CRS to {'init': 'epsg:4326'}
birds.crs = {'init' :'epsg:4326'}

# Check your answer
q_1.check()
''',
 11: '''# Your code here
ax = americas.plot(figsize=(10,10), color='white', linestyle=':', edgecolor='gray')
birds.plot(ax=ax, markersize=10)

# Uncomment to zoom in
#ax.set_xlim([-110, -30])
#ax.set_ylim([-30, 60])

# Uncomment to see a hint
#q_2.hint()
''',
 16: '''# Your code here
# Take the last recorded position of each tagged bird.
end_df = birds.groupby("tag-local-identifier")['geometry'].apply(list).apply(lambda x: x[-1]).reset_index()
end_gdf = gpd.GeoDataFrame(end_df, geometry=end_df.geometry)
end_gdf.crs = {'init': 'epsg:4326'}

# Check your answer
q_3.check()
''',
 19: '''# Your code here
ax = americas.plot(figsize=(10, 10), color='white', linestyle=':', edgecolor='gray')

start_gdf.plot(ax=ax, color='red',  markersize=30)
path_gdf.plot(ax=ax, cmap='tab20b', linestyle='-', linewidth=1, zorder=1)
end_gdf.plot(ax=ax, color='black', markersize=30)

# Uncomment to see a hint
#q_4.hint()
''',
 32: '''# Your code here: Calculate the total area of South America (in square kilometers)
# EPSG 3035 is an equal-area projection, so areas computed in it are metres
# squared and comparable; dividing by 10**6 gives square kilometres.
totalArea = sum(south_america.geometry.to_crs(epsg=3035).area) / 10**6

# Check your answer
q_7.check()
''',
 37: '''# Your code here
ax = south_america.plot(figsize=(10,10), color='white', edgecolor='gray')
protected_areas[protected_areas['MARINE']!='2'].plot(ax=ax, alpha=0.4, zorder=1)
birds[birds.geometry.y < 0].plot(ax=ax, color='red', alpha=0.6, markersize=10, zorder=2)

# Uncomment to see a hint
#q_8.hint()
''',
},
"exercise-proximity-analysis.ipynb": {
 16: '''# Your code here
# A 10 km buffer around every hospital, merged into one shape; a collision is
# out of range if that shape does not contain it.
coverage = gpd.GeoDataFrame(geometry=hospitals.geometry).buffer(10000)
my_union = coverage.geometry.unary_union
outside_range = collisions.loc[~collisions["geometry"].apply(lambda x: my_union.contains(x))]

# Check your answer
q_3.check()
''',
 21: '''def best_hospital(collision_location):
    # Your code here
    idx_min = hospitals.geometry.distance(collision_location).idxmin()
    my_hospital = hospitals.iloc[idx_min]
    name = my_hospital["name"]
    return name

# Test your function: this should suggest CALVARY HOSPITAL INC
print(best_hospital(outside_range.geometry.iloc[0]))

# Check your answer
q_4.check()
''',
 24: '''# Your code here
highest_demand = outside_range.geometry.apply(best_hospital).value_counts().idxmax()

# Check your answer
q_5.check()
''',
 29: None,   # filled by substituting the four blanks; see LATLONG below
},
}

# ex3 and ex4 are long map-building cells; keep them in their own dicts for
# readability rather than inlining above.
CELLS["exercise-interactive-maps.ipynb"] = {
 10: None, 15: None, 24: None, 29: None,
}
CELLS["exercise-manipulating-geospatial-data.ipynb"] = {
 10: None, 13: None, 22: None, 27: None, 32: None, 35: None,
}


def replace_stub(text, replacement):
    """Swap the single `____` line for real code, keeping the rest of the cell."""
    return text.replace("____", replacement, 1)


# The proximity-analysis hospital cell has four blanks and a long map-building
# body. Rewriting it whole once cost both: q_6.check() became
# q_6.check(new_percentage), which raises TypeError because Q6 declares
# _var = "new_percentage" and takes no argument, and the folium map, markers and
# heatmap were dropped entirely. The bare `except:` swallowed the TypeError and
# called hint(), so the question was silently never graded. Substitute the
# blanks in place instead, and leave every other line of the cell alone.
LATLONG = ["40.6714", "-73.8492", "40.6702", "-73.7612"]


# For the two map-heavy notebooks the answer is always a drop-in for the lone
# `____`, so express them as the replacement text only.
INLINE = {
"exercise-interactive-maps.ipynb": {
 10: "HeatMap(data=earthquakes[['Latitude', 'Longitude']], radius=15).add_to(m_1)",
 15: """def color_producer(val):
    if val < 50:
        return 'forestgreen'
    elif val < 100:
        return 'darkorange'
    else:
        return 'darkred'

for i in range(0,len(earthquakes)):
    folium.Circle(
        location=[earthquakes.iloc[i]['Latitude'], earthquakes.iloc[i]['Longitude']],
        radius=2000,
        color=color_producer(earthquakes.iloc[i]['Depth'])).add_to(m_2)""",
 24: """Choropleth(geo_data=prefectures['geometry'].__geo_interface__,
           data=stats['density'],
           key_on="feature.id",
           fill_color='YlGnBu',
           legend_name='Population density (per square kilometer)'
          ).add_to(m_3)""",
 29: """def color_producer(magnitude):
    if magnitude > 6.5:
        return 'red'
    else:
        return 'green'

Choropleth(
    geo_data=prefectures['geometry'].__geo_interface__,
    data=stats['density'],
    key_on="feature.id",
    fill_color='BuPu',
    legend_name='Population density (per square kilometer)').add_to(m_4)

for i in range(0,len(earthquakes)):
    folium.Circle(
        location=[earthquakes.iloc[i]['Latitude'], earthquakes.iloc[i]['Longitude']],
        popup=("{} ({})").format(
            earthquakes.iloc[i]['Magnitude'],
            earthquakes.iloc[i]['DateTime'].year),
        radius=earthquakes.iloc[i]['Magnitude']**5.5,
        color=color_producer(earthquakes.iloc[i]['Magnitude'])).add_to(m_4)""",
},
"exercise-coordinate-reference-systems.ipynb": {
 22: "protected_areas = gpd.read_file(protected_filepath)",
 25: """ax = south_america.plot(figsize=(10,10), color='white', edgecolor='gray')
protected_areas.plot(ax=ax, alpha=0.4)""",
},
"exercise-proximity-analysis.ipynb": {
 8: "HeatMap(data=collisions[['LATITUDE', 'LONGITUDE']], radius=9).add_to(m_1)",
 13: """for idx, row in hospitals.iterrows():
    Marker([row['latitude'], row['longitude']], popup=row['name']).add_to(m_2)""",
},
"exercise-manipulating-geospatial-data.ipynb": {
 10: """def my_geocoder(row):
    point = geolocator.geocode(row).point
    return pd.Series({'Latitude': point.latitude, 'Longitude': point.longitude})

berkeley_locations = rows_with_missing.apply(lambda x: my_geocoder(x['Address']), axis=1)
starbucks.update(berkeley_locations)""",
 13: """for idx, row in starbucks[starbucks["City"]=='Berkeley'].iterrows():
    Marker([row['Latitude'], row['Longitude']]).add_to(m_2)""",
 22: """cols_to_add = CA_pop.join([CA_high_earners, CA_median_age]).reset_index()
CA_stats = CA_counties.merge(cols_to_add, on="GEOID")""",
 27: """sel_counties = CA_stats[((CA_stats.high_earners > 100000) &
                         (CA_stats.median_age < 38.5) &
                         (CA_stats.density > 285) &
                         ((CA_stats.median_age < 35.5) |
                         (CA_stats.density > 1400) |
                         (CA_stats.high_earners > 500000)))]""",
 32: """len(gpd.sjoin(starbucks_gdf, sel_counties))""",
 35: """mc = MarkerCluster()

locations_of_interest = gpd.sjoin(starbucks_gdf, sel_counties)
for idx, row in locations_of_interest.iterrows():
    if not math.isnan(row['Longitude']) and not math.isnan(row['Latitude']):
        mc.add_child(folium.Marker([row['Latitude'], row['Longitude']]))

m_6.add_child(mc)""",
},
}

FILES = {
 "ex1": "exercise-your-first-map.ipynb",
 "ex2": "exercise-coordinate-reference-systems.ipynb",
 "ex3": "exercise-interactive-maps.ipynb",
 "ex4": "exercise-manipulating-geospatial-data.ipynb",
 "ex5": "exercise-proximity-analysis.ipynb",
}


def build(name, filename):
    nb = json.loads((HERE / "sources" / filename).read_text())
    full = {k: v for k, v in CELLS.get(filename, {}).items() if v is not None}
    inline = INLINE.get(filename, {})
    touched = 0

    # The hospital cell has four blanks wrapped around a long map-building body.
    # Rewriting it whole once broke two things at once: q_6.check() became
    # q_6.check(new_percentage), which raises TypeError because Q6 declares
    # _var = "new_percentage" and takes no argument, and the folium map, markers
    # and heatmap were dropped. The bare `except:` swallowed the TypeError and
    # called hint(), so the question was silently never graded. Substitute the
    # blanks in place and leave every other line alone.
    if filename == "exercise-proximity-analysis.ipynb":
        cell = nb["cells"][29]
        text = "".join(cell["source"])
        if text.count("____") != 4:
            raise SystemExit("proximity-analysis cell 29: expected 4 blanks")
        for value in LATLONG:
            text = text.replace("____", value, 1)
        cell["source"] = text.splitlines(keepends=True)
        cell["outputs"] = []
        cell["execution_count"] = None
        touched += 1

    for index in sorted(set(full) | set(inline)):
        cell = nb["cells"][index]
        text = "".join(cell["source"])
        if "____" not in text:
            raise SystemExit(f"{filename}: cell {index} has no ____ to fill")
        new = full[index] if index in full else replace_stub(text, inline[index])
        cell["source"] = new.splitlines(keepends=True)
        cell["outputs"] = []
        cell["execution_count"] = None
        touched += 1
    out = HERE / name / filename
    out.write_text(json.dumps(nb, indent=1) + "\n")
    left = sum(1 for c in nb["cells"]
               if c["cell_type"] == "code" and "____" in "".join(c["source"]))
    print(f"wrote {out.relative_to(HERE.parents[1])}  ({touched} cells, {left} ____ left)")


def main():
    for name, filename in FILES.items():
        build(name, filename)


if __name__ == "__main__":
    main()
