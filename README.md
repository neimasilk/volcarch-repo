# VOLCARCH 🌋🏛️

**Volcanic Taphonomic Bias in Indonesian Archaeological Records**

> Can we computationally predict where ancient civilizations in volcanic Java are buried underground?

## The Problem

Java has been one of the most densely populated regions on Earth for millennia. It also sits on one of the most active volcanic arcs in the world. Every few decades, eruptions from volcanoes like Kelud, Semeru, and Arjuno deposit centimeters of ash across the landscape. Over centuries, this buries archaeological sites meters underground — making them invisible to conventional survey.

Meanwhile, the "oldest" known kingdom in Indonesia (Kutai, ~400 CE) is in Kalimantan — a region with **zero active volcanoes**. Coincidence?

## The Approach

We use machine learning, GIS, and volcanological data to build a predictive model with two layers:

1. **Settlement Suitability Model** — Where would ancient people have lived? (ML trained on known sites + environmental features)
2. **Volcanic Burial Depth Model** — How deep is the ancient surface now buried? (Volcanic eruption history + tephra dispersal modeling)

The overlay produces a probability map: *"Here is where settlements likely existed, buried at approximately this depth."*

## Status

🔬 **Phase 1: Computational Foundation** (in progress)

See `docs/L2_STRATEGY.md` for current research activities.

## Structure

```
docs/           Research documents (layered PRD + journal)
data/           Raw and processed datasets
experiments/    Numbered, self-contained experiments
models/         Trained ML models
maps/           Generated probability maps
papers/         Paper drafts
tools/          Shared utility scripts
```

## Empirical Anchor

Our framework is calibrated against the Dwarapala statues of Singosari (Malang, East Java) — giant 3.7m stone guardians from ~1268 CE that were found with half their height buried in the 19th century, yielding a measurable sedimentation rate of ~3.6 mm/year.

## Contributing

This is an academic research project. If you are an archaeologist, geologist, or GIS specialist interested in collaboration, please open an issue or contact us.

## License

Code: MIT. Papers and documents: CC BY 4.0. Data: see individual source licenses.
