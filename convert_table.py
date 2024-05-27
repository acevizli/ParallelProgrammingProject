import json

def json_to_latex_table(json_file, latex_file):

    with open(json_file, 'r') as file:
        data = json.load(file)
    with open(latex_file, 'w') as file:
        file.write("\\begin{table}[h]\n")
        file.write("\\centering\n")
        file.write("\\begin{tabular}{|l|r|r|r|}\n")
        file.write("\\hline\n")
        file.write("Benchmark & Time (ms) & CPU (ms) & Iterations \\\\\n")
        file.write("\\hline\n")
        
        for bench in data['benchmarks']:
            name = bench['name']
            time = f"{bench.get('real_time', 0):.2f}"
            cpu = f"{bench.get('cpu_time', 0):.2f}"
            iterations = bench.get('iterations', 'N/A')
            
            file.write(f"{name} & {time} & {cpu} & {iterations} \\\\\n")
        
        file.write("\\hline\n")
        file.write("\\end{tabular}\n")
        file.write("\\caption{Benchmark Results}\n")
        file.write("\\label{tab:benchmark_results}\n")
        file.write("\\end{table}\n")

# Example usage
json_to_latex_table("./build/results.json", "benchmark_results.tex")