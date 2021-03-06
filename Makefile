.PHONY : build clean gh-action lint sync tests

build : lint tests
	python setup.py sdist
	twine check dist/*.tar.gz
	# Check that we can install the packaged version.
	pip install dist/*.tar.gz

lint :
	flake8

tests :
	# Touch cython files to ensure they get recompiled.
	touch tests/*.pyx
	# Step into the test directory to make sure we don't accidentally import directly from cyfunc.
	cd tests && pytest -v

sync : requirements.txt
	pip-sync

requirements.txt : requirements.in setup.py
	pip-compile -v

clean :
	rm -rf build *.egg-info dist
	rm -f cyfunc/*.c cyfunc/*.so

# Build the repository using a GitHub action for local debugging (cf. https://github.com/nektos/act).
gh-action :
	act -P ubuntu-latest=nektos/act-environments-ubuntu:18.04
