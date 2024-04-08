.PHONY: install clean lint format 

## Install for production
install:
	@echo ">> Installing dependencies"
	@uv pip install --upgrade pip
	@uv pip install -e .

## Install for development 
install-dev: install
	@uv pip install -e ".[dev]"

## Delete all temporary files
clean:
	@rm -rf .ipynb_checkpoints
	@rm -rf **/.ipynb_checkpoints
	@rm -rf .pytest_cache
	@rm -rf **/.pytest_cache
	@rm -rf __pycache__
	@rm -rf **/__pycache__
	@rm -rf build
	@rm -rf dist

## Format files using black
format:
	@ruff check . --fix
	@ruff format .


## Run checks (ruff + test)
check:
	@ruff check .
	@ruff format --check .

## Build using pip-tools
build:
	@uv pip install --upgrade pip
	@uv pip install --upgrade pip-tools
	@uv pip compile pyproject.toml -o requirements.txt
	@uv pip compile pyproject.toml --extra dev -o requirements-dev.txt


## Prepare data
run:
	litestar run --port 8000

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available commands:$$(tput sgr0)"
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
