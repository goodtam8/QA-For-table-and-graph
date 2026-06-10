from verifier import verify
from parser import main as run_pdf_job
from qa import answerdirectly


def main() -> None:
    # convert to markdown
    run_pdf_job()
    # verify
    check = verify()
    if check == True:
        while True:
            question = input("Ask a question about the HSBC document: ")
            answer = answerdirectly(question)
    
            print(answer)
    else:
        print("false")


if __name__ == "__main__":
    main()