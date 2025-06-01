# Hardware Files for AI-Powered Electronic Nose

This folder contains everything needed to reproduce the hardware build:

## 1. Schematics
- **Location:** `hardware/schematics/`
- **Contents:**
  - Native CAD files (e.g., `stm32_k210_e_nose.sch` or KiCad project files)
  - PDF/PNG exports of the schematic (`stm32_k210_e_nose_schematic.pdf`, etc.)

## 2. PCB
- **Location:** `hardware/pcb/`
- **Contents:**
  - Gerber package (e.g., `stm32_k210_e_nose_gerbers.zip`)
  - Board-layer image exports (e.g., `board_top.png`, `board_bottom.png`)
  - Any PDF exports showing top/bottom copper layers (e.g., `.pdf`)

## 3. Bill of Materials (BOM)
- **Location:** `hardware/bills_of_materials/`
- **Contents:**
  - `BOM.csv` — a comma-separated list of every component, reference designator, value, footprint, cost, and supplier link.
  - `README.md` — instructions and notes on how to interpret or update the BOM.

## 4. Enclosure (Optional)
- **Location:** `hardware/enclosure/`
- **Contents:**
  - 3D files for the enclosure (e.g., `enclosure.stl`, STEP files)
  - Assembly notes (`README.md`) for how to print or assemble the enclosure.

---

**Notes & Tips:**
- Keep your CAD sources (e.g., `.sch`, `.kicad_pcb`) versioned so others can modify or review.
- Ensure Gerber outputs exactly match the PCB images (double-check layer order before exporting).
- Update `BOM.csv` whenever you add/remove a component—include exact part numbers and footprints.
- If you add an enclosure, include a short text explanation in `hardware/enclosure/README.md` for print settings (e.g., material, layer height).