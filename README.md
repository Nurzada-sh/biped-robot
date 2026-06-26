# biped-robot
<img width="1545" height="1623" alt="image" src="https://github.com/user-attachments/assets/80da62fd-5e92-4cac-966f-a1d280631691" />
<img width="1048" height="1467" alt="image" src="https://github.com/user-attachments/assets/93c9a482-f8eb-40b5-858b-26c2fcbb524e" />
<img width="1048" height="1467" alt="image" src="https://github.com/user-attachments/assets/57685736-5ef8-4f70-856f-7a071e77237e" />



# Исследование и разработка программного модуля для идентификации динамических параметров исполнительных механизмов гибридного робота с шагающими колесами, используемых при обучении политик управления с подкреплением (RL)

**Автор:** Шумкарбек кызы Нурзада, группа R4133c(507206)

## Цель проекта
Разработать программный модуль для идентификации динамических параметров исполнительных механизмов гибридного робота с шагающими колесами, используемых при обучении политик управления с подкреплением (RL).

## Описание
Программный модуль для идентификации динамических параметров исполнительных механизмов.

## Функциональность
- Разработка модели гибридного робота с шагающими колесами в формате MJCF (MuJoCo)
- Идентификации динамических параметров исполнительных механизмов с помощью метода DREM(Dynamic Regressor Extension and Mixing)
- Синтез идентифицированных параметров в политику обучения с подкреплением(SAC)
- Валидация параметров модели

## Структура проекта
- config.py – parameters
- core.py – DREM and environment
- collect.py – data collection and identification
- train.py – training and testing
- main.py – CLI entry point
- requirements.txt - ависимости
- LICENSE MIT лицензия
- README.md - документация

## Технологии
- Python 3.8+
- 
- MuJoCo (mjpro 2.3+)
  
## Установка
```bash
git clone https://github.com/Nurzada-sh/biped-robot.git
cd 
pip install -r requirements.txt
