from __future__ import annotations


def test_public_dataset_generation_cli_imports():
    from scripts.generate_dataset import main

    assert callable(main)


def test_public_pastis_conversion_cli_imports():
    from scripts.convert_pastis_dataset import main

    assert callable(main)


def test_public_dacia5_conversion_cli_imports():
    from scripts.convert_dacia5_dataset import main

    assert callable(main)


def test_public_unet_training_cli_imports():
    from scripts.train_unet import main

    assert callable(main)


def test_public_random_forest_training_cli_imports():
    from scripts.train_random_forest import main

    assert callable(main)


def test_public_evaluation_cli_imports():
    from scripts.evaluate import main

    assert callable(main)

def test_public_prediction_cli_imports():
    from scripts.predict import main

    assert callable(main)

def test_scripts_package_imports():
    import scripts

    assert scripts is not None