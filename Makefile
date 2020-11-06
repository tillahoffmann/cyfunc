.PHONY : build clean gh-action lint sync tests

build : lint tests

lint :
	flake8

tests :
	# Step into the test directory to make sure we don't accidentally import directly from cyfunc.
	cd tests && pytest -v

sync : requirements.txt
	pip-sync

requirements.txt : requirements.in setup.py
	pip-compile -v

clean :
	rm -rf build *.egg-info
	rm -f cyfunc/*.c cyfunc/*.so
	touch tests/*.pyx

# Build the repository using a GitHub action for local debugging (cf. https://github.com/nektos/act).
gh-action :
	act -P ubuntu-latest=nektos/act-environments-ubuntu:18.04
