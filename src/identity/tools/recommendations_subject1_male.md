# Pairing recommendations for `subject1_male`

_Updated 2026-05-16. Subject gender: **male**. Cosine similarity to subject1 embedding (ArcFace buffalo_l, averaged across 3 real videos)._

## Top 2 selected identities

### DFM swap models (DeepFaceLive)

Cosine computed on **post-swap embedding** — subject1's face is run through the .dfm model, then the result is re-embedded.

| #1 (dfm1) | #2 (dfm2) |
| --- | --- |
| <img src="previews/Bryan_Greynolds.png" width="240" alt="Bryan Greynolds"/> | <img src="previews/Tim_Norland.png" width="240" alt="Tim Norland"/> |
| **Bryan Greynolds**<br/>cosine 0.0129, far, male | **Tim Norland**<br/>cosine −0.0074, far, male |

### FF identities (FaceFusion / inswapper)

Cosine computed on **source image embedding** (averaged across all available source photos per identity).

| #1 (ff1) | #2 (ff2) |
| --- | --- |
| <img src="sources/Tom_Holland1.jpg" width="240" alt="Tom Holland"/> | <img src="sources/Ted_Mosby1.jpg" width="240" alt="Ted Mosby (Josh Radnor)"/> |
| **Tom Holland**<br/>cosine 0.0002, far, male | **Ted Mosby (Josh Radnor)**<br/>cosine −0.0423, far, male |
