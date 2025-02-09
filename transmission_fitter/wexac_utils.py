import os

def generate_lsf_file(
    job_name,
    queue,
    output_file,
    error_file,
    command,
    wall_time="01:00",  # Default wall time is 1 hour
    num_cores=1,
    memory="2GB",
    lsf_file_name="job.lsf"):
    """
    Generate an .lsf file for batch submission.

    Parameters:
        job_name (str): Name of the job.
        queue (str): Queue name to submit the job.
        output_file (str): File to store the standard output.
        error_file (str): File to store the standard error.
        command (str): Command to execute in the job.
        wall_time (str): Wall time for the job (e.g., "01:00" for 1 hour).
        num_cores (int): Number of CPU cores to request.
        memory (str): Memory request (e.g., "4GB").
        lsf_file_name (str): Name of the .lsf file to generate.
    """
    # Create the .lsf file content
    lsf_content = f"""#BSUB -J {job_name}
    #BSUB -q {queue}
    #BSUB -W {wall_time}
    #BSUB -R "affinity[thread*4]"
    #BSUB -R "rusage[mem={memory}]"
    #BSUB -o {output_file}
    #BSUB -e {error_file}

    # Command to execute
    {command}
    """

    # Write the content to the .lsf file
    with open(lsf_file_name, "w") as file:
        file.write(lsf_content)

    print(f"LSF file '{lsf_file_name}' has been created.")

# Example usage
if __name__ == "__main__":
    # Set job parameters
    job_name = "example_job"
    queue = "short"
    output_file = "stdout.%J"
    error_file = "stderr.%J"
    command = "python script.py"
    wall_time = "02:00"  # 2 hours
    num_thread = 1
    memory = "2GB"
    lsf_file_name = "example_job.lsf"

    # Generate the .lsf file
    generate_lsf_file(
        job_name=job_name,
        queue=queue,
        output_file=output_file,
        error_file=error_file,
        command=command,
        wall_time=wall_time,
        num_cores=num_cores,
        memory=memory,
        lsf_file_name=lsf_file_name
    )