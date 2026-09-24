# EXAMGUARD AI — COMPREHENSIVE DATASET GUIDE

## 1. Overview & Privacy Principles
ExamGuard AI operates in academic and high-stakes examination environments. Training datasets must adhere strictly to academic integrity guidelines, ethical biometric handling, and data privacy regulations (GDPR / FERPA).

## 2. Required Classes & Visual Attributes
| Class ID | Label | Key Visual Characteristics | Hard Examples |
| :--- | :--- | :--- | :--- |
| `0` | `person` | Upper torso, shoulders, neck, head | Second person peering from background; candidate shifting off-screen |
| `1` | `cell_phone` | Rectangular form factor, glass reflection, active illuminated display | Wallets, power banks, dark calculators |
| `2` | `laptop` | Clamshell open base and LCD screen | External keyboard, closed tablet, desktop monitor edge |
| `3` | `book` | Bound spine, rectangular paper edges, printed text textures | Blank paper sheets, passport, mousepad |
| `4` | `earphone` | Earbud enclosure, in-ear stem, cord / wire | Hearing aids, ear accessories, hair covering ears |

## 3. Dataset Splitting & Augmentation
- **Train Split (65-70%)**: High visual variability across lighting conditions, camera focal lengths, and indoor backgrounds.
- **Validation Split (15-20%)**: Unseen candidate sessions for checkpoint selection (`best.pt`).
- **Test Split (10-15%)**: Real-world evaluation split for generating final benchmark reports.

### Augmentations Applied (via `configs/training.yaml`)
- Horizontal Flip ($p=0.5$)
- Hue, Saturation, Value perturbation ($\pm 1.5\% / \pm 50\% / \pm 30\%$)
- Scale jitter ($\pm 15\%$)
- Mosaic augmentation ($p=0.5$)

## 4. Minimum Recommended Dataset Sizes
- **Fine-Tuning from Pre-Trained YOLOv8**: $\ge 1,500$ annotated frames across diverse candidates.
- **Industrial Deployment Baseline**: $\ge 10,000$ multi-demographic frames covering varied lighting, skin tones, and camera resolutions ($720p$ / $1080p$).
