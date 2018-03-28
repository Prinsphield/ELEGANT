# ELEGANT: Exchanging Latent Encodings with GAN for Transferring Multiple Face Attributes

By Taihong Xiao, Jiapeng Hong and Jinwen Ma

## Test

### 1. Swap Attribute

With

    python ELEGANT.py -m test -a Bangs Mustache -r 34000 --swap --swap_list 1 --input ./images/goodfellow_aligned.png --target ./images/bengio_aligned.png

<div align="center">
<img align="center" src="extra/swap.jpg" alt="swap">
</div>
<div align="center">
Swap Attribute
</div>
<br/>


### 2. Linear Interpolation

    python ELEGANT.py -m test -a Bangs Mustache -r 34000 --linear --swap_list 1 --input ./images/bengio_aligned.png --target ./images/goodfellow_aligned.png -s 4

### 3. Matrix Interpolation with Respect to One Attribute

    python ELEGANT.py -m test -a Bangs Mustache -r 34000 --matrix --swap_list 0 --input ./images/ng_aligned.png --target ./images/bengio_aligned.png ./images/goodfellow_aligned.png ./images/jian_sun_aligned.png -s 4 4

### 4. Matrix Interpolation with Respect to Two Attributes

    python ELEGANT.py -m test -a Bangs Mustache -r 34000 --matrix --swap_list 0 1 --input ./images/lecun_aligned.png --target ./images/bengio_aligned.png ./images/goodfellow_aligned.png -s 4 4
