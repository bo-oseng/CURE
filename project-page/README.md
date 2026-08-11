# CURE Project Page

Static project page for **CURE: Controllable Unified Image Restoration for Complex Degradations** (ICPR 2026).

## Preview locally

From this directory:

```bash
python3 -m http.server 8000
```

Then open `http://localhost:8000`.

## Deploy with GitHub Pages

The directory is self-contained and has no build step. The repository's
`.github/workflows/pages.yml` publishes this directory whenever `main` changes.
In **Settings → Pages**, set **Source** to **GitHub Actions** once to enable the
deployment.

The canonical URL and Open Graph URLs in `index.html` currently target:

```text
https://bo-oseng.github.io/CURE/
```

Update those values if the final repository name or domain changes.

## Content map

- `index.html`: paper copy, metadata, page structure, links, and BibTeX
- `static/css/index.css`: visual system and responsive layout
- `static/js/index.js`: ratio control, selective restoration, order comparison, supplementary explorer/lightbox, and BibTeX copy
- `static/images/ratio/`: Figure 3(a) outputs for `w = 0.2 ... 0.9`
- `static/images/selective/`: selective restoration examples
- `static/images/order/`: restoration-order comparison
- `static/images/qualitative/`: representative CCDD-11 results
- `static/images/results/figure-02.jpg`: full Figure 2 qualitative comparison extracted at 300 dpi
- `static/images/method/`: CURE overview figure
- `static/images/supplement/`: optimized supplementary assets; Figures 5–19 are presented in the page explorer
- `static/pdfs/CURE.pdf`: local paper PDF

The interactive ratio demo intentionally uses the eight values visualized in the paper. Replace or extend these files when `w = 0.0, 0.1, 1.0` outputs are available.

The supplementary explorer groups fifteen figures into Identity Preservation, Ratio Control, Selective Control, Order Dependency, and Restoration Quality. Contextual buttons in the main narrative open the corresponding group and figure. Figure 4 (real-world restoration) is intentionally omitted from the page.

## Credits

Based on the [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template), which is licensed under CC BY-SA 4.0.
