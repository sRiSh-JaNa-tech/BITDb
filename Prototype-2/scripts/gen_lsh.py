import numpy as np

def generate_lsh_vectors(output_path):
    np.random.seed(42) # fixed seed for reproducibility
    # 24 hyperplanes, 384 dimensions each
    vectors = np.random.randn(24, 384).astype(np.float32)
    
    with open(output_path, "w") as f:
        f.write("#pragma once\n\n")
        f.write("#include <vector>\n\n")
        f.write("static const float LSH_HYPERPLANES[24][384] = {\n")
        for i in range(24):
            f.write("    {\n        ")
            for j in range(384):
                f.write(f"{vectors[i, j]:.6f}f")
                if j < 383:
                    f.write(", ")
                if (j + 1) % 8 == 0 and j < 383:
                    f.write("\n        ")
            f.write("\n    }")
            if i < 23:
                f.write(",")
            f.write("\n")
        f.write("};\n")

if __name__ == "__main__":
    generate_lsh_vectors("C:/Users/srish/Desktop/BitDB/Prototype-2/src/lsh_vectors.h")
