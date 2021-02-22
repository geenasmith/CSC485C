# Optimization Ideas

* CV_32F for output of sobel as a packed value
    * may have limited input if just used for output

* try with/without input as 8 bit int
* attempt to find way to reduce output size
* if input is 8 bit pixels, can we still somehow do in place?
    * find max/min value to write in place
