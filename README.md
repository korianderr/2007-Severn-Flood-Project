Flood Risk Model - River Severn

This project builds a flood risk model for the River Severn, around Tewkesbury and Gloucester. It uses the July 2007 Severn floods as a case study.

The goal is to estimate how much flood damage can be expected in a typical year, and how bad a rare flood could be.

What it does
- A simple flood model generates many example floods at different river levels.
- A CNN (a type of neural network) is trained on these examples. It learns to predict flood depth from the terrain and the river level, much faster than running the flood model each time.
- Fifty years of real river level data is used to work out how often different flood sizes happen.
- This is used to simulate 10,000 possible years of flooding.
- Each simulated year is turned into a flood map using the neural network.
- Building data from OpenStreetMap is used to work out how much damage each flood would cause.
- All of this together gives an estimate of expected flood damage per year, and a chance of larger, rarer floods.

Main result

Expected flood damage per year: about £36 million. This ranges from £31 million to £57 million, depending on some reasonable modelling choices explained below.

This is compared to £6.33 billion worth of buildings that could be affected by flooding in this area.

How the project is laid out
solver.py - the simple flood model
generate_scenarios.py - creates training data from the flood model
model.py, train.py - the CNN and its training
evaluate.py - checks how accurate the CNN is
pot_analysis.py - works out how often floods of different sizes happen
monte_carlo.py - simulates 10,000 years of flooding
exposure.py - gets building data and lines it up with the map
damage.py - turns flood depth into money lost

Things worth knowing

The CNN is accurate, but not perfect. It gets the total flooded area right to within 0.5%, when compared to the real flood model. It gets individual pixels right about 82% of the time. This is because it sometimes predicts flooding in slightly the wrong place, but these mistakes usually cancel out when you add up the total area.

We could not fully work out how rare the 2007 flood was. Normal statistical methods gave an unstable answer, because there are 50 years of data but only one very large flood (2007) in that time. There just isn't enough data to be sure. Because of this, the model uses three different reasonable assumptions, and the 2007 flood's estimated return period ranges from about 1-in-23 years to about 1-in-576 years.

Most of the damage comes from fairly common floods, not rare ones. Only about 6% of the expected yearly damage comes from floods rarer than 1-in-100 years. This is a bit different to how flood risk is usually described, but it comes directly from how often floods happen in the real data.

The CNN is slower than the simple flood model it is based on. This is because the simple flood model is already very fast. The CNN is still useful here because it only needs to be run a small number of times (about 90), instead of once for every single one of the 10,000 simulated years.

Main limitations
- No flood defences are included. Both towns have real flood defences, so this model likely overestimates damage.
- Very small, frequent floods are not included in the model, so this may underestimate damage.
- Building values are a flat assumed number, not real property values.
- The damage model (how much money is lost per metre of water) is simplified.
- The uncertainty shown only covers the three main assumptions above. There are other uncertainties not shown.

How to run it

pip install -r requirements.txt

python src/generate_scenarios.py
python src/train.py
python src/evaluate.py
python src/exposure.py
python src/monte_carlo.py
python src/damage.py

Data used
Terrain: Environment Agency LIDAR data
River level: Environment Agency Hydrology API, Haw Bridge gauge, 1975-2026
Buildings: OpenStreetMap