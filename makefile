MAKE_RECURSIVE_DIRS := paper_my_stress paper_copy_attention
run:
	cd overleaf-toolkit && bin/up
generate: generate_paper
	bash -c "$${MAKE_RECURSIVE}"
generate_paper:
	find . -maxdepth 1 -type d -name 'paper*' | tail +2 | xargs -IX sh -c "cp -r paper/. X/"
define MAKE_RECURSIVE
if [ -n "$${MAKE_RECURSIVE_PARALLEL}" ]; then
	trap 'kill 0' EXIT INT TERM
	time printf '%s\n' $(MAKE_RECURSIVE_DIRS) | xargs -P0 -IX sh -c 'cd X && $(MAKE) $@'
	wait
else
	time printf '%s\n' $(MAKE_RECURSIVE_DIRS) | xargs -IX sh -c 'cd X && $(MAKE) $@'
fi
endef
export