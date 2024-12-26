(TeX-add-style-hook
 "cate-vignette"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("mathpazo" "sc") ("fontenc" "T1") ("natbib" "authoryear")))
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art10"
    "mathpazo"
    "fontenc"
    "geometry"
    "url"
    "amsmath"
    "natbib"
    "authblk")
   (LaTeX-add-labels
    "sec:intro"
    "sec:cate"
    "sec:factor")
   (LaTeX-add-bibliographies
    "ref"))
 :latex)

