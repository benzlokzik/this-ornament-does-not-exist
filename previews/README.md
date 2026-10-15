# Option matrix (seed 7)

Every discrete combination of the `orna generate` flags at seed 7, rendered at 400×600 and shown at 320×480. Columns are the five palettes; each row is one set of flags. Reproduce a cell with:

```shell
orna generate --seed 7 <row flags> --palette <column> --width 400 --height 600 --single --output out/tile.png
```

| flags | `botanica` | `moonlit` | `ember` | `verdigris` | `plum-mist` |
| --- | --- | --- | --- | --- | --- |
| `--preset botanica` | ![botanica botanica](matrix/botanica--botanica.png) | ![botanica moonlit](matrix/botanica--moonlit.png) | ![botanica ember](matrix/botanica--ember.png) | ![botanica verdigris](matrix/botanica--verdigris.png) | ![botanica plum-mist](matrix/botanica--plum-mist.png) |
| `--preset botanica --no-opening` | ![botanica-no-opening botanica](matrix/botanica-no-opening--botanica.png) | ![botanica-no-opening moonlit](matrix/botanica-no-opening--moonlit.png) | ![botanica-no-opening ember](matrix/botanica-no-opening--ember.png) | ![botanica-no-opening verdigris](matrix/botanica-no-opening--verdigris.png) | ![botanica-no-opening plum-mist](matrix/botanica-no-opening--plum-mist.png) |
| `--preset botanica --density 2.5` | ![botanica-dense botanica](matrix/botanica-dense--botanica.png) | ![botanica-dense moonlit](matrix/botanica-dense--moonlit.png) | ![botanica-dense ember](matrix/botanica-dense--ember.png) | ![botanica-dense verdigris](matrix/botanica-dense--verdigris.png) | ![botanica-dense plum-mist](matrix/botanica-dense--plum-mist.png) |
| `--preset botanica --no-opening --density 2.5` | ![botanica-no-opening-dense botanica](matrix/botanica-no-opening-dense--botanica.png) | ![botanica-no-opening-dense moonlit](matrix/botanica-no-opening-dense--moonlit.png) | ![botanica-no-opening-dense ember](matrix/botanica-no-opening-dense--ember.png) | ![botanica-no-opening-dense verdigris](matrix/botanica-no-opening-dense--verdigris.png) | ![botanica-no-opening-dense plum-mist](matrix/botanica-no-opening-dense--plum-mist.png) |
| `--preset bloom` | ![bloom botanica](matrix/bloom--botanica.png) | ![bloom moonlit](matrix/bloom--moonlit.png) | ![bloom ember](matrix/bloom--ember.png) | ![bloom verdigris](matrix/bloom--verdigris.png) | ![bloom plum-mist](matrix/bloom--plum-mist.png) |
| `--preset ribbons` | ![ribbons botanica](matrix/ribbons--botanica.png) | ![ribbons moonlit](matrix/ribbons--moonlit.png) | ![ribbons ember](matrix/ribbons--ember.png) | ![ribbons verdigris](matrix/ribbons--verdigris.png) | ![ribbons plum-mist](matrix/ribbons--plum-mist.png) |

A palette changes only the colours; the composition for a given seed is identical across palettes. `--no-opening` removes the moonlit landscape entirely.

## Gallery with seed 7

## Galleries by palette (seed 7)

`orna gallery --seed 7 --palette <palette> --width 1200 --single --output out/gallery-<palette>.png`, shown at 800 px wide. The seed fixes every motif; only the colours change between columns.

| `botanica` | `moonlit` | `ember` | `verdigris` | `plum-mist` |
| --- | --- | --- | --- | --- |
| ![gallery botanica](galleries/gallery-botanica.png) | ![gallery moonlit](galleries/gallery-moonlit.png) | ![gallery ember](galleries/gallery-ember.png) | ![gallery verdigris](galleries/gallery-verdigris.png) | ![gallery plum-mist](galleries/gallery-plum-mist.png) |
