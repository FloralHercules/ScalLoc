# External data only

Download **Thermal-Oblique** separately and place trajectory folders under
`Thermal-Oblique/test/changsha/`. Put the satellite map under
`maps/changsha/ref.tif`. The default fast pipeline does not read
`maps/changsha/dsm.tif`, so downloading or copying a DSM is unnecessary.
Keep the `paths.dsm_map` configuration field as a compatibility placeholder;
the corresponding file may be absent.

Alternatively point `configs/local.yaml` to existing files without copying them.
See [the complete asset guide](../docs/DATA_AND_WEIGHTS.md).
No dataset, satellite image, DSM, image sample or data symlink is shipped here.
