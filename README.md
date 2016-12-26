# vid-stitch
Python program to help stitch individual videos from BlackVue dashcams (like the DR650S), into a single clip importable into Final Cut Pro X.
# Usage
```commandline
python findframes.py ~/Movies/DashCam/20160925_09*F.mp4
```

After processing, will write a file in /tmp/ that you can import into FCPX.

# Gotchas
* Sometimes BlackVue cameras skip frames between videos entirely. If this happens, the two clips will just play right after the other (we don't insert blank frames or anything.)
* Lower-framerate clips (with N in their name) confuse OpenCV (which we use to process images). You may not get good results if you have these.

