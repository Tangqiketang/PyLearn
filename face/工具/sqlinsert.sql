CREATE TABLE `t_person` (
  `id` varchar(32) NOT NULL,
  `person_name` varchar(32) NOT NULL,
  `face_embedding` blob,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

调用
## 将python返回的flaot[],使用tool工具类生成十六进制数据

INSERT INTO t_person (id, person_name, face_embedding)
VALUES (
  '1',
  '王敏',
  UNHEX('ssss'));