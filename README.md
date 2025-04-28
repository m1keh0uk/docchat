# DocChat

DocChat is a smart chatbot that allows users to ask questions about documents (HTML files, PDFs, webpages, or images).  


![Tests](https://github.com/m1keh0uk/docchat/actions/workflows/tests.yml/badge.svg)

---

## Demo

![Demo](media/example_gif.gif)

Example:
```markdown

## Requirements

Requires textract, requests, beautifulsoup4, groq, python-dotenv
```
$ pip3 install -r requirements.txt
```

## Example

```
$ python docchat.py decleration.txt

docchat> Whats the most common theme?
DOCCHAT: The most common theme in the Declaration of Independence is the idea of Natural Rights and the right to self-governance. The document asserts that all men are created equal, endowed with certain unalienable rights such as life, liberty, and the pursuit of happiness, and that governments derive their just powers from the consent of the governed. It also emphasizes the need to alter or abolish a government that becomes destructive of these rights, and the right of the people to throw off such a government and establish a new one.

docchat> How does the rocket engine work underwater?
DOCCHAT: Sorry, this document does not contain information about underwater rocket engines.
```