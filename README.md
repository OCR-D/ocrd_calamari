# ocrd_calamari

Recognize text using [Calamari OCR](https://github.com/Calamari-OCR/calamari).

Introduction
-------------

This offers a OCR-D compliant workspace processor for some of the functionality of Calamari OCR.

This processor only operates on the text line level and so needs a line segmentation (and by extension a binarized 
image) as its input.

Example Usage
-------------

~~~
ocrd-calamari-recognize -p test-parameters.json -m mets.xml -I OCR-D-SEG-LINE -O OCR-D-OCR-CALAMARI
~~~

With `test-parameters.json`:
~~~
{
    "checkpoint": "/path/to/some/trained/models/*.ckpt.json"
}
~~~
