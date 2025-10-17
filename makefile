run:
	cd overleaf-toolkit && bin/up
generate:
	MSYS_NO_PATHCONV=1 docker run --rm -it -v $(shell realpath paper):/out -w /out -e OUT=main paperist/alpine-texlive-ja bash -c "$${paper}"
define paper
sed -i -e "/^.*\\doi.*/d;" $${OUT}.bib
sed -i -e "s/、/，/g;s/。/．/g" $${OUT}.tex
lualatex $${OUT}
upbibtex $${OUT}
lualatex $${OUT}
lualatex $${OUT}
endef
export