# CalciumHandling

This is an attempt at analyzing calcium fluorescence peaks using linescans 

What to do (still trying to discover it)
1.Prepare csv files with 2 columns with frame data and fluorescense intensity (the ones you obtain from the imageJ macro in Calcium friend
2.Run the python script using the following prompts in bash:
If single file:
python peak_analysis_app.py \
  -i path/to/your/data.csv \
  -o ./results \
  --frame_rate 500 \
  --smooth_window 15 \
  --smooth_poly 3
If all files in directory:
python peak_analysis_app.py \
  -i path/to/csv_folder \
  -o ./results \
  --frame_rate 500 \
  --smooth_window 15 \
  --smooth_poly 3
3.Iterative peak detection in which you will be prompted to choose height threshold (minimum peak intensity), minimum distance between peaks and if you would like to skip the first and/or the last peak. (good standard to start is using heigh=100 and distance=300(frames))
4. From all the peaks detected, a mean BPM will be calculated which will be used to create a local window to the first peak of each file (unless skipped) where you can add and subtract seconds to the window to include the full peak
5. Everything should be calculated and ready to go

Additional tips
-Frame rate, smoothing window and polyorder can be adjusted to match your acquisition system.

-Use a larger smoothing window/polynomial to reduce noise, but not so large that you blunt real peaks.
