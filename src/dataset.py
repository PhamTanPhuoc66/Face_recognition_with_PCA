
import os
import random
import numpy as np

# Khai báo một class có nhiệm vụ quản lý dữ liệu ảnh 

class DatasetClass:

    def __init__(self, required_no=None, train_percentage=None, random_sampling=False, 
                 max_images_per_person=20, train_images_per_person=6): 
        """
        Initialize DatasetClass with different splitting modes
        
        Args:
            required_no (int): Fixed number of images for training per person (legacy mode)
            train_percentage (float): Percentage of images to use for training (0.0 to 1.0)
            random_sampling (bool): If True, randomly sample max_images_per_person and split
            max_images_per_person (int): Maximum images to sample per person when random_sampling=True
            train_images_per_person (int): Number of training images per person when random_sampling=True
        """

        # Get absolute path to datasets directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.dir = os.path.join(current_dir, "datasets") # Đường dẫn đến thư mục chứa ảnh
        
        # Verify the datasets directory exists
        if not os.path.exists(self.dir):
            print(f"Warning: Datasets directory does not exist: {self.dir}")
            print("Please make sure you have created the datasets folder and added person folders with images.")
        else:
            print(f"Using datasets directory: {self.dir}")
            print(f"Found folders: {os.listdir(self.dir) if os.path.isdir(self.dir) else 'None'}")
        
        # Initialize variables for splitting mode
        self.train_percentage = train_percentage
        self.required_no = required_no
        self.random_sampling = random_sampling
        self.max_images_per_person = max_images_per_person
        self.train_images_per_person = train_images_per_person
        
        # Determine which mode to use
        if random_sampling:
            self.use_random_sampling = True
            print(f"Using random sampling: {max_images_per_person} images per person, {train_images_per_person} for training, {max_images_per_person - train_images_per_person} for testing")
        elif train_percentage is not None:
            self.use_percentage_split = True
            self.use_random_sampling = False
            print(f"Using percentage-based split: {train_percentage*100:.1f}% training, {(1-train_percentage)*100:.1f}% testing")
        else:
            self.use_percentage_split = False
            self.use_random_sampling = False
            print(f"Using fixed number split: {required_no} images for training per person")
        
        # Khởi tạo các biến để lưu trữ thông tin về ảnh và nhãn
        self.images_name_for_train = [] # Danh sách đường dẫn ảnh train
        self.y_for_train = []  # Danh sách nhãn tương ứng ảnh train
        self.no_of_elements_for_train = [] # là một danh sách, mỗi phần tử là số lượng ảnh train của người tương ứng.

        self.target_name_as_array= [] # Danh sách tên người tương ứng với nhãn
        self.label_to_name_dict = {} # Từ điển ánh xạ từ nhãn → tên người

        self.images_name_for_test = [] # Danh sách đường dẫn ảnh test
        self.y_for_test = [] # Danh sách nhãn tương ứng ảnh test
        self.no_of_elements_for_test = [] # Số ảnh test mỗi người

        # Process the dataset
        self._process_dataset()
    
    def _process_dataset(self):
        """Process the dataset and split into train/test based on the chosen mode"""
        person_id = 0 # Biến đếm số người (thư mục con) đã xử lý
        
        for name in os.listdir(self.dir):
            dir_path = os.path.join(self.dir, name)
            if os.path.isdir(dir_path):
                # Get all image files in the directory
                image_files = [img_name for img_name in os.listdir(dir_path) 
                              if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                
                total_images = len(image_files)
                
                if self.use_random_sampling:
                    # Random sampling mode: exactly max_images_per_person images per person
                    if total_images < self.max_images_per_person:
                        print(f"Skipping person '{name}': Only {total_images} images, need at least {self.max_images_per_person}")
                        continue
                    
                    # Randomly sample exactly max_images_per_person images
                    random.seed(42)  # For reproducible results
                    selected_images = random.sample(image_files, self.max_images_per_person)
                    
                    # Split into train/test
                    train_images = selected_images[:self.train_images_per_person]
                    test_images = selected_images[self.train_images_per_person:]
                    
                    print(f"Person '{name}': {total_images} total, sampled {self.max_images_per_person}, {len(train_images)} for training, {len(test_images)} for testing")
                    
                    # Process training images
                    for i, img_name in enumerate(train_images):
                        img_path = os.path.join(dir_path, img_name)
                        self.images_name_for_train.append(img_path)
                        self.y_for_train.append(person_id)
                        
                        if len(self.no_of_elements_for_train) > person_id:
                            self.no_of_elements_for_train[person_id] += 1
                        else:
                            self.no_of_elements_for_train.append(1)
                        
                        if i == 0:  # First image for this person
                            self.target_name_as_array.append(name)
                            self.label_to_name_dict[person_id] = name
                    
                    # Process testing images
                    for img_name in test_images:
                        img_path = os.path.join(dir_path, img_name)
                        self.images_name_for_test.append(img_path)
                        self.y_for_test.append(person_id)
                        
                        if len(self.no_of_elements_for_test) > person_id:
                            self.no_of_elements_for_test[person_id] += 1
                        else:
                            self.no_of_elements_for_test.append(1)
                    
                    person_id += 1
                    
                else:
                    # Original logic for percentage or fixed number splits
                    # Calculate the number of training images based on mode
                    if self.use_percentage_split:
                        train_count = max(1, int(total_images * self.train_percentage))  # At least 1 for training
                    else:
                        train_count = min(self.required_no, total_images)  # Don't exceed available images
                    
                    # Ensure we have enough images to split
                    if total_images >= train_count:
                        print(f"Person '{name}': {total_images} total images, {train_count} for training, {total_images - train_count} for testing")
                        
                        # Process images for this person
                        for i, img_name in enumerate(image_files):
                            img_path = os.path.join(dir_path, img_name)

                            if i < train_count:
                                # Ảnh cho tập train
                                self.images_name_for_train.append(img_path)
                                self.y_for_train.append(person_id)

                                # Theo dõi số lượng ảnh huấn luyện của từng người
                                if len(self.no_of_elements_for_train) > person_id:
                                    self.no_of_elements_for_train[person_id] += 1 
                                else:
                                    self.no_of_elements_for_train.append(1)
                                
                                # Lưu tên người tương ứng với nhãn số (chỉ lần đầu)
                                if i == 0:
                                    self.target_name_as_array.append(name)
                                    self.label_to_name_dict[person_id] = name

                            else:
                                # Ảnh cho tập test
                                self.images_name_for_test.append(img_path)
                                self.y_for_test.append(person_id)

                                # Theo dõi số lượng ảnh test của từng người
                                if len(self.no_of_elements_for_test) > person_id:
                                    self.no_of_elements_for_test[person_id] += 1
                                else:
                                    self.no_of_elements_for_test.append(1)

                        person_id += 1
                    else:
                        print(f"Skipping person '{name}': Only {total_images} images, need at least {train_count} for training")
        
        # Print summary
        total_train = len(self.images_name_for_train)
        total_test = len(self.images_name_for_test)
        total_images = total_train + total_test
        
        print(f"\n=== Dataset Split Summary ===")
        print(f"Total persons: {len(self.target_name_as_array)}")
        print(f"Total images: {total_images}")
        print(f"Training images: {total_train} ({total_train/total_images*100:.1f}%)")
        print(f"Testing images: {total_test} ({total_test/total_images*100:.1f}%)")
        print(f"Person names: {self.target_name_as_array}")
        
        if self.use_random_sampling:
            print(f"Random sampling: {self.max_images_per_person} images per person, {self.train_images_per_person} train, {self.max_images_per_person - self.train_images_per_person} test")

