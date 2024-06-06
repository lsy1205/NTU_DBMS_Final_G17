CREATE DATABASE  IF NOT EXISTS DBMSFinal;
USE DBMSFinal;

-- Example Transaction Table --
create table transact(
	id INT PRIMARY KEY auto_increment,
	step INT,
    amount REAL,
    oldbalanceOrg REAL,
	newbalanceOrig REAL,
    oldbalanceDest REAL,
    newbalanceDest REAL,
    orig_diff INT,
    dest_diff INT,
    surge INT,
    freq_dest INT,
    true_type INT
);

-- Example Model Table --
create table Model(
	id INT PRIMARY KEY AUTO_INCREMENT,
	epochs INT,
    learning_rate REAL,
    momentum REAL
);

-- Example Configuration Table --
CREATE TABLE config(
	id INT PRIMARY KEY AUTO_INCREMENT,
    threshold REAL
);

-- Example Result Table --
CREATE TABLE result(
	id INT PRIMARY KEY AUTO_INCREMENT,
    action_type ENUM("Model", "Threshold"),
    message VARCHAR(100)
);
INSERT INTO result(action_type, message) 
VALUES("Model", "Default Model"),
("Threshold", "Default Threshold"); 

SELECT * FROM result;

-- Example Transaction Data --
INSERT INTO transact(step, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, orig_diff, dest_diff, surge, freq_dest, true_type) 
VALUES
	(9, 73400.6, 109293.09, 35892.49, 3195858.1, 3351554.23, 0, 1, 0, 0, 1),
	(7, 135.88, 107519.0, 107383.12, 0.0, 0.0, 0, 1, 0, 0, 3),
    (1, 181.0, 181.0, 0.0, 0.0, 0.0, 0, 1, 0, 0, 0);
    
SELECT * FROM transact;

-- Example MODEL Hyperparameters --
INSERT INTO Model(epochs, learning_rate, momentum) VALUE(50, 0.01,0.9);
SELECT * FROM Model;

Update Model
SET epochs = 20, learning_rate = 0.0001, momentum = 0.9
WHERE id = 1;

-- Example CONFIGURATION Parameters
INSERT INTO config(threshold) VALUE(0.6);
SELECT * FROM config;
UPDATE config
SET threshold = 1.0
WHERE id = 1;

-- Example OOD Detection Function --
SELECT id, 
my_function(step, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, 
newbalanceDest, orig_diff, dest_diff, surge, freq_dest, true_type) as is_ood 
FROM transact;

-- Example Create Model Function --
UPDATE result
SET message = create_model(
    (SELECT epochs FROM Model where id = 1),
    (SELECT learning_rate FROM Model where id = 1),
    (SELECT momentum FROM Model where id = 1)
)
WHERE id = 1;
SELECT * FROM result;

-- Example Change Threshold Function --
UPDATE result
SET message = change_threshold((SELECT(threshold) FROM config WHERE id = 1))
WHERE id = 2;
SELECT * FROM result;

-- DROP TABLES & DATABASES --
DROP TABLE transact;
DROP TABLE Model;
DROP TABLE config;
DROP TABLE result;
DROP DATABASE DBMSFinal;
