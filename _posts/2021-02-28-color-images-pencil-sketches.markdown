---
layout: post
title: "Converting Color Images to Black and White Pencil Sketches using OpenCV"
date: 2021-02-28
---

<p class="intro"><span class="dropcap">I</span>n this post we will implement a small opencv function to convert color images into black and white pencil sketches.</p>
<p>In order to obtain the pencil sketches of the images, we will use two image-blending techniques called <strong>Dodging</strong> and <strong>Burning</strong>.</p>
* Dodging
<p>lightens the image by decreasing the exposure of the image to light.</p>
* Burning
<p>darkens the image by increasing the exposure of the image to light.</p>
<p>In Image processing to obtain dodging and burning we use mask, Mask is an array of same dimesions as image. Assume Mask as a paper with hole and control exposure for a specific portion of an image by letting the light as depicted below in the figure.</p>
<figure>
	<img src="{{ '/assets/img/dodge-burn.jpg' | prepend: site.baseurl }}" alt=""> 
	<figcaption>Fig1. - Dodging and Burning</figcaption>
</figure>
If you search on the Internet, you might stumble upon the following common procedure to achieve a pencil sketch from an RGB color image:

1. Convert the color image to grayscale.
2. Invert the grayscale image to get a negative.
3. Apply a Gaussian blur to the negative from step 2.
4. Blend the grayscale image from step 1 with the blurred negative from step 3 using a color dodge.

#### Step 1:

Lets import the necessary opencv libraries and read the color image and convert it to grayscale image.
{%- highlight python -%}
#=> importing the opencv library
import cv2
img = cv2.imread('./images/portrait.jpg', cv2.IMREAD_COLOR)
#=> opencv function to obtain grayscale of color image
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

#=> Checking if the image file was read properly or not
if img is None:
print('No Image found!')

{%- endhighlight -%}

#### Step 2:

In this step we will convert the grayscale image to negative by inverting every pixel of the image.
{%- highlight python -%}
#=> grayscale to negative, pixel values range from 0 to 255
img_gray_inv = 255 - img_gray
{%- endhighlight -%}

#### Step 3:

A Gaussian blur is basically a convolution with a Gaussian function. It is an effective way to both reduce noise and reduce the amount of detail in an image (also called smoothing an image).

{%- highlight python -%}
#=> Applying Gaussian Blur
blurred_image = cv2.GaussianBlur(img_gray, (21, 21), 0, 0)
{%- endhighlight -%}

#### Step 4:

Now we will combine the image of 1 ( grayscale) and the step 4 Gaussian blur and
{%- highlight python -%}
#=> Pixelwise division
gray_sketch = cv2.divide(img_gray, blurred_image, scale=256)
#=> If we want to ouput as if we have drawn on canvas
if self.canvas is not None:
gray_sketch = cv2.multiply(gray_sketch, self.canvas, scale=1 / 256)
img_pencil = cv2.cvtColor(gray_sketch, cv2.COLOR_GRAY2RGB)
{%- endhighlight -%}

Thas it! Lets display the image we have obtained
{%- highlight python -%}
#=> Display the Pencil Sketch Image
cv2.imshow('Test image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
{%- endhighlight -%}

<figure>
	<img src="{{ '/assets/img/portrait.jpg' | prepend: site.baseurl }}" alt=""> 
	<figcaption>Fig2. - Original Image</figcaption>
</figure>
<figure>
<img src="{{ '/assets/img/pencil_sketch.jpg' | prepend: site.baseurl }}" alt=""> 
	<figcaption>Fig2. - Pencil Sketch</figcaption>
</figure>

Check out the full [Source Code][github]

## References

- [Creating Black and White Pencil Sketches][packt]

[github]: https://github.com/Vikas-KM/opencv-learning/tree/main/chap-01
[packt]: https://subscription.packtpub.com/book/application_development/9781785282690/1/ch01lvl1sec10/creating-a-black-and-white-pencil-sketch
