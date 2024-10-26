import random
import math
import matplotlib.pyplot as plt

# ฟังก์ชันพื้นฐาน
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def matrix_multiply(a, b):
    result = [[sum(x * y for x, y in zip(row, col)) for col in zip(*b)] for row in a]
    return result

def scale_features(data):
    # Standardization
    scaled_data = []
    for i in range(len(data[0])):
        col = [row[i] for row in data]
        mean = sum(col) / len(col)
        std_dev = math.sqrt(sum((x - mean) ** 2 for x in col) / len(col))
        scaled_data.append([(x - mean) / std_dev for x in col])
    return list(map(list, zip(*scaled_data)))

# อ่านข้อมูลจากไฟล์ wdbc.txt
def load_data(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            if len(parts) == 32:  # เช็คจำนวนคอลัมน์
                id_num = parts[0]
                diagnosis = 1 if parts[1] == 'M' else 0  # แปลง M = 1, B = 0
                features = list(map(float, parts[2:]))   # แปลงฟีเจอร์เป็น float
                data.append(features + [diagnosis])      # รวมฟีเจอร์และ class
    return data

# Class สำหรับ MLP
class MLP:
    def __init__(self, input_size, hidden_size, output_size, weights):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # น้ำหนักแบ่งเป็นสองชุด: input -> hidden และ hidden -> output
        self.weights_input_hidden = weights[:input_size * hidden_size]
        self.weights_hidden_output = weights[input_size * hidden_size:]

    def forward(self, inputs):
        # คำนวณ hidden layer
        hidden_layer_input = [sum(inputs[j] * self.weights_input_hidden[j + i * self.input_size] 
                                  for j in range(self.input_size)) for i in range(self.hidden_size)]
        hidden_layer_output = [sigmoid(x) for x in hidden_layer_input]

        # คำนวณ output layer
        output_layer_input = [sum(hidden_layer_output[j] * self.weights_hidden_output[j + i * self.hidden_size]
                                  for j in range(self.hidden_size)) for i in range(self.output_size)]
        output = [sigmoid(x) for x in output_layer_input]
        
        return output

# ฟังก์ชันการคำนวณ fitness
def calculate_fitness(mlp, data):
    correct_predictions = 0
    for inputs, target in data:
        prediction = mlp.forward(inputs)[0]
        correct_predictions += int((prediction > 0.5) == target)
    return correct_predictions / len(data)

# ฟังก์ชันสำหรับการ crossover
def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

# ฟังก์ชันสำหรับ mutation
def mutate(weights, mutation_rate):
    for i in range(len(weights)):
        if random.random() < mutation_rate:
            weights[i] += random.uniform(-0.1, 0.1)  # ปรับค่าของน้ำหนัก
    return weights

# ฟังก์ชัน GA สำหรับการค้นหาน้ำหนักที่เหมาะสม
def genetic_algorithm(data, input_size, hidden_size, output_size, population_size, generations, mutation_rate):
    population = [[random.uniform(-1, 1) for _ in range(input_size * hidden_size + hidden_size * output_size)] for _ in range(population_size)]

    for generation in range(generations):
        fitness_scores = []
        
        # คำนวณ fitness ของแต่ละโครโมโซม
        for weights in population:
            mlp = MLP(input_size, hidden_size, output_size, weights)
            fitness = calculate_fitness(mlp, data)
            fitness_scores.append((fitness, weights))
        
        # เลือกโครโมโซมที่ดีที่สุด
        fitness_scores.sort(reverse=True, key=lambda x: x[0])
        population = [weights for _, weights in fitness_scores[:population_size // 2]]
        
        # ทำ crossover และ mutation เพื่อสร้างประชากรใหม่
        new_population = []
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(population, 2)
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1, mutation_rate))
            new_population.append(mutate(child2, mutation_rate))
        
        population = new_population
        print(f"Generation {generation + 1}, Best Fitness: {fitness_scores[0][0]:.4f}")

    # return โครโมโซมที่มี fitness ดีที่สุด
    return fitness_scores[0][1]

# ฟังก์ชันสำหรับ 10% cross-validation
import matplotlib.pyplot as plt

# ฟังก์ชันสำหรับ 10% cross-validation ที่เพิ่มการแสดงกราฟ
def cross_validation(data, input_size, hidden_size, output_size, population_size, generations, mutation_rate, k=10):
    fold_size = len(data) // k
    accuracies = []

    for i in range(k):
        # แบ่งข้อมูลสำหรับการทดสอบ (1 fold) และการเทรน (9 folds)
        test_data = data[i * fold_size:(i + 1) * fold_size]
        train_data = data[:i * fold_size] + data[(i + 1) * fold_size:]

        # หา weight ที่ดีที่สุดด้วย Genetic Algorithm
        best_weights = genetic_algorithm(train_data, input_size, hidden_size, output_size, population_size, generations, mutation_rate)

        # ทดสอบความแม่นยำของโมเดลบน test data
        mlp = MLP(input_size, hidden_size, output_size, best_weights)
        accuracy = calculate_fitness(mlp, test_data)
        accuracies.append(accuracy)
        print(f"Fold {i + 1}, Test Accuracy: {accuracy:.4f}")

    # ค่าเฉลี่ยของความแม่นยำจากทุก fold
    average_accuracy = sum(accuracies) / k
    print(f"Average Accuracy after {k}-fold cross-validation: {average_accuracy:.4f}")

    # แสดงผลกราฟ
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, k + 1), accuracies, marker='o', linestyle='-', color='b', label='Accuracy per Fold', markersize=8)
    plt.axhline(y=average_accuracy, color='r', linestyle='--', label='Average Accuracy')

    # เพิ่มการแสดงค่า accuracy ในแต่ละ fold บนกราฟ
    for i, acc in enumerate(accuracies):
        plt.text(i + 1, acc + 0.01, f'{acc:.2f}', ha='center', fontsize=10)

    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.title(f'Cross-Validation Accuracy (k={k})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return average_accuracy


# โหลดและเตรียมข้อมูล
data = load_data('wdbc.txt')

# Scaling ข้อมูล
X = [row[:-1] for row in data]  # ดึงข้อมูลฟีเจอร์
y = [row[-1] for row in data]   # ดึงข้อมูล class
X = scale_features(X)           # Standardize ข้อมูล

# เตรียมข้อมูลสำหรับ GA
train_data = list(zip(X, y))
input_size = len(X[0])
hidden_size = 30  # กำหนดจำนวนโหนดใน hidden layer
output_size = 1

# ใช้งาน cross-validation
cross_validation(train_data, input_size, hidden_size, output_size, population_size=50, generations=20, mutation_rate=0.1, k=10)
