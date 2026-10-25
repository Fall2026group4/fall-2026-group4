# Word report

The formal written document is built in LaTeX (`reports/Latex_report/main.tex`).
The course also asks for a Word version; this folder holds it.

Plan: when `main.tex` has its Methods and Results sections (Instructor Review #6 asked for these by
2026-11-10), export it to `SAE-Faithful_Report.docx` with

    pandoc reports/Latex_report/main.tex --citeproc --bibliography=reports/Latex_report/mybib.bib -o reports/Word_Report/SAE-Faithful_Report.docx

and check the figures and tables by eye. Keep the two versions in sync in the same pull request.
