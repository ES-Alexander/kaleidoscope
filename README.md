# kaleidoscope
A kaleidoscope effect for images and videos.

Implements the functionality displayed in 
[this GIMP tutorial](https://www.youtube.com/watch?v=C6Y9Yh4BM1Q), plus some extra
configurability.   
Test image is taken from that tutorial video for comparison.

## Requirements
Functionality requires numpy for kaleidoscope creation, and opencv for display and
annotation   
(can use `pip install -r requirements.txt`, `pip3` on Mac).

## Running
Run `kaleidoscope.py` as a command-line script (`-h` for help with parameters), or
import the `kaleido` function to your script and pass in an image and the relevant
parameters as desired.

## Example
Run `interactive.py` (`-h` for help with parameters), and modify the sliders to
change the kaleidoscope parameters.
![interactive example](https://github.com/ES-Alexander/kaleidoscope/blob/main/interactive_example.png?raw=true)


### TODO
- Add [pythonic-cv](https://github.com/ES-Alexander/pythonic-cv) to enable an interactive
kaleidoscope effect in video playback.
