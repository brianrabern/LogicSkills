CREATE TABLE `names` (
  `symbol` VARCHAR(56) PRIMARY KEY,
  `name` VARCHAR(100),
  `gender` ENUM('male', 'female', 'other') DEFAULT 'other',
  `time_created` INT(11) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE `predicates` (
  `symbol` VARCHAR(56) PRIMARY KEY,
  `template` VARCHAR(255),
  `negated_template` VARCHAR(255),
  `arity` TINYINT(1),
  `structure` VARCHAR(56),
  `semantic_type` VARCHAR(56),
  `tense` VARCHAR(56),
  `time_created` INT(11) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE `sentences` (
  `id` INT AUTO_INCREMENT PRIMARY KEY,
  `sentence` TEXT DEFAULT NULL,
  `type` VARCHAR(56) DEFAULT NULL,
  `soa` JSON DEFAULT NULL,
  `form` JSON DEFAULT NULL,
  `ast` JSON DEFAULT NULL,
  `base` TINYINT(1) DEFAULT NULL,
  `status` TINYINT(1) DEFAULT NULL,
  `time_created` INT(11) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
