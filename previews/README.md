# Option matrix (seed 7)

Every discrete combination of the `orna generate` flags at seed 7, rendered at 320×480. Columns are the six palettes; each row is one set of flags. Reproduce a cell with:

```shell
orna generate --seed 7 <row flags> --palette <column> --width 320 --height 480 --single --output out/tile.png
```

A palette changes only the colours; the composition for a given seed is identical across palettes. The tiles and gallery sheets here are reduced to 256 colours to keep the repository small; the renders themselves are full colour.

## `botanica`: landscape × background

`--landscape` paints the moonlit landscape openly across the cool region (`open`, default), inside a framed window (`window`), or not at all (`none`). `--background` fills the free space with overlapping scrolls, leaves and small flowers (`dense`, default), with a field of flowers of varied species and hues (`blooms`), or with sparse sprigs of leaves (`sprigs`).

| flags | `botanica` | `moonlit` | `ember` | `verdigris` | `plum-mist` | `nocturne` |
| --- | --- | --- | --- | --- | --- | --- |
| `--landscape open --background dense` | ![open dense botanica](matrix/botanica-open-dense--botanica.png) | ![open dense moonlit](matrix/botanica-open-dense--moonlit.png) | ![open dense ember](matrix/botanica-open-dense--ember.png) | ![open dense verdigris](matrix/botanica-open-dense--verdigris.png) | ![open dense plum-mist](matrix/botanica-open-dense--plum-mist.png) | ![open dense nocturne](matrix/botanica-open-dense--nocturne.png) |
| `--landscape open --background blooms` | ![open blooms botanica](matrix/botanica-open-blooms--botanica.png) | ![open blooms moonlit](matrix/botanica-open-blooms--moonlit.png) | ![open blooms ember](matrix/botanica-open-blooms--ember.png) | ![open blooms verdigris](matrix/botanica-open-blooms--verdigris.png) | ![open blooms plum-mist](matrix/botanica-open-blooms--plum-mist.png) | ![open blooms nocturne](matrix/botanica-open-blooms--nocturne.png) |
| `--landscape open --background sprigs` | ![open sprigs botanica](matrix/botanica-open-sprigs--botanica.png) | ![open sprigs moonlit](matrix/botanica-open-sprigs--moonlit.png) | ![open sprigs ember](matrix/botanica-open-sprigs--ember.png) | ![open sprigs verdigris](matrix/botanica-open-sprigs--verdigris.png) | ![open sprigs plum-mist](matrix/botanica-open-sprigs--plum-mist.png) | ![open sprigs nocturne](matrix/botanica-open-sprigs--nocturne.png) |
| `--landscape window --background dense` | ![window dense botanica](matrix/botanica-window-dense--botanica.png) | ![window dense moonlit](matrix/botanica-window-dense--moonlit.png) | ![window dense ember](matrix/botanica-window-dense--ember.png) | ![window dense verdigris](matrix/botanica-window-dense--verdigris.png) | ![window dense plum-mist](matrix/botanica-window-dense--plum-mist.png) | ![window dense nocturne](matrix/botanica-window-dense--nocturne.png) |
| `--landscape window --background blooms` | ![window blooms botanica](matrix/botanica-window-blooms--botanica.png) | ![window blooms moonlit](matrix/botanica-window-blooms--moonlit.png) | ![window blooms ember](matrix/botanica-window-blooms--ember.png) | ![window blooms verdigris](matrix/botanica-window-blooms--verdigris.png) | ![window blooms plum-mist](matrix/botanica-window-blooms--plum-mist.png) | ![window blooms nocturne](matrix/botanica-window-blooms--nocturne.png) |
| `--landscape window --background sprigs` | ![window sprigs botanica](matrix/botanica-window-sprigs--botanica.png) | ![window sprigs moonlit](matrix/botanica-window-sprigs--moonlit.png) | ![window sprigs ember](matrix/botanica-window-sprigs--ember.png) | ![window sprigs verdigris](matrix/botanica-window-sprigs--verdigris.png) | ![window sprigs plum-mist](matrix/botanica-window-sprigs--plum-mist.png) | ![window sprigs nocturne](matrix/botanica-window-sprigs--nocturne.png) |
| `--landscape none --background dense` | ![none dense botanica](matrix/botanica-none-dense--botanica.png) | ![none dense moonlit](matrix/botanica-none-dense--moonlit.png) | ![none dense ember](matrix/botanica-none-dense--ember.png) | ![none dense verdigris](matrix/botanica-none-dense--verdigris.png) | ![none dense plum-mist](matrix/botanica-none-dense--plum-mist.png) | ![none dense nocturne](matrix/botanica-none-dense--nocturne.png) |
| `--landscape none --background blooms` | ![none blooms botanica](matrix/botanica-none-blooms--botanica.png) | ![none blooms moonlit](matrix/botanica-none-blooms--moonlit.png) | ![none blooms ember](matrix/botanica-none-blooms--ember.png) | ![none blooms verdigris](matrix/botanica-none-blooms--verdigris.png) | ![none blooms plum-mist](matrix/botanica-none-blooms--plum-mist.png) | ![none blooms nocturne](matrix/botanica-none-blooms--nocturne.png) |
| `--landscape none --background sprigs` | ![none sprigs botanica](matrix/botanica-none-sprigs--botanica.png) | ![none sprigs moonlit](matrix/botanica-none-sprigs--moonlit.png) | ![none sprigs ember](matrix/botanica-none-sprigs--ember.png) | ![none sprigs verdigris](matrix/botanica-none-sprigs--verdigris.png) | ![none sprigs plum-mist](matrix/botanica-none-sprigs--plum-mist.png) | ![none sprigs nocturne](matrix/botanica-none-sprigs--nocturne.png) |

## Density and the other presets

`--density` multiplies the number of filler elements (background ornament, vine flowers, plant branching); `bloom` and `ribbons` ignore the landscape and background flags.

| flags | `botanica` | `moonlit` | `ember` | `verdigris` | `plum-mist` | `nocturne` |
| --- | --- | --- | --- | --- | --- | --- |
| `--preset botanica --density 2.5` | ![density botanica](matrix/botanica-density25--botanica.png) | ![density moonlit](matrix/botanica-density25--moonlit.png) | ![density ember](matrix/botanica-density25--ember.png) | ![density verdigris](matrix/botanica-density25--verdigris.png) | ![density plum-mist](matrix/botanica-density25--plum-mist.png) | ![density nocturne](matrix/botanica-density25--nocturne.png) |
| `--preset bloom` | ![bloom botanica](matrix/bloom--botanica.png) | ![bloom moonlit](matrix/bloom--moonlit.png) | ![bloom ember](matrix/bloom--ember.png) | ![bloom verdigris](matrix/bloom--verdigris.png) | ![bloom plum-mist](matrix/bloom--plum-mist.png) | ![bloom nocturne](matrix/bloom--nocturne.png) |
| `--preset ribbons` | ![ribbons botanica](matrix/ribbons--botanica.png) | ![ribbons moonlit](matrix/ribbons--moonlit.png) | ![ribbons ember](matrix/ribbons--ember.png) | ![ribbons verdigris](matrix/ribbons--verdigris.png) | ![ribbons plum-mist](matrix/ribbons--plum-mist.png) | ![ribbons nocturne](matrix/ribbons--nocturne.png) |

## Galleries by palette (seed 7)

`orna gallery --seed 7 --palette <palette> --width 800 --single --output out/gallery-<palette>.png`. The seed fixes every motif; only the colours change between columns. Sections: ribbons, spirals, scrolls, flowers, foliage, plants, palette swatches.

| `botanica` | `moonlit` | `ember` | `verdigris` | `plum-mist` | `nocturne` |
| --- | --- | --- | --- | --- | --- |
| ![gallery botanica](galleries/gallery-botanica.png) | ![gallery moonlit](galleries/gallery-moonlit.png) | ![gallery ember](galleries/gallery-ember.png) | ![gallery verdigris](galleries/gallery-verdigris.png) | ![gallery plum-mist](galleries/gallery-plum-mist.png) | ![gallery nocturne](galleries/gallery-nocturne.png) |
