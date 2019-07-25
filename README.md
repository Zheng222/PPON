# PPON
Pytorch implemention of "Progressive Perception-Oriented Network for Single Image Super-Resolution"

[[arXiv]](https://arxiv.org/abs/1907.10399)

<p align="center">
    <img src="figures/Structure.png" width="480"> <br />
    <em> The schematics of the Progressive Perception-Oriented Network</em>
</p>

<p align="center">
    <img src="figures/show.jpg" width="960"> <br />
    <em> The example results</em>
</p>

## PI VS LPIPS
![PI-VS-LPIPS](figures/PI%20VS%20LPIPS.png)
As illustrated in the above picture, we can obviously see that the PI score of EPSR3 (2.2666) is even better than HR (2.3885), but EPSR3 shows unnatural and lacks proper texture and structure.

## Example Results
![Perceptual-results-1](figures/example.png)
![Perceptual-results-2](figures/example2.png)
