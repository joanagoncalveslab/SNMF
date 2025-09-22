


#
#! Data Paths

# Get current working directory
current_dir = os.getcwd()
# Move up to the project root
project_root = os.path.abspath(os.path.join(current_dir, "../.."))  # goes up 3 levels


N_train = 100
N_test = 1000
fold = '_all'

# Construct train paths
path_X_train = os.path.join(project_root, "data", "processed", "bootstrapped_sameSplit_sorted", f"N_{N_train}", f"X_train{fold}.text")
path_Y_train = os.path.join(project_root, "data", "processed", "bootstrapped_sameSplit_sorted", f"N_{N_train}", f"Y_train{fold}.text")

# Construct test paths
path_X_test = os.path.join(project_root, "data", "processed", "bootstrapped_sameSplit_sorted", f"N_{N_test}", f"X_test{fold}.text")
path_Y_test = os.path.join(project_root, "data", "processed", "bootstrapped_sameSplit_sorted", f"N_{N_test}", f"Y_test{fold}.text")


