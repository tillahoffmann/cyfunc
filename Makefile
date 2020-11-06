sync : requirements.txt
	pip-sync

requirements.txt : setup.py requirements.in
	pip-compile -v

clean :
	rm -rf build *.egg-info
	rm -f cyfunc/*.c cyfunc/*.so

tests : clean
	# Ensure any changes are picked up
	pip install -e .
	touch tests/*.pyx
	pytest -v
