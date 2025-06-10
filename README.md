# cv-depth-analyser – monocular object-distance estimation

This tool turns a single webcam into an interactive tool that *finds an object you name and overlays its metric distance in real time*!

1. **Install**
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Run**
   ```bash
   python3 main.py
   ```
3. **Calibrate**

   If needed, calibrate the tool based on your webcams FOV and resolution. Use a standard 6x9 checkerboard for this.
