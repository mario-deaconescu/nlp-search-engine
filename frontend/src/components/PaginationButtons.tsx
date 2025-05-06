import {Button} from "@heroui/react";

export const PaginationButtons = ({currentPage, setCurrentPage, numPages}:{
    currentPage: number,
    setCurrentPage: (page: number) => void,
    numPages: number,
}) => {
    const goToPage = (page: number) => {
        if (page >= 1 && page <= numPages) {
            setCurrentPage(page);
        }
    }

    return (
        <div className="mt-4 flex items-center gap-4 ">
            <Button
                variant={"solid"}
                color={"primary"}
                radius={'sm'}
                onPress={() => goToPage(currentPage - 1)}
                isDisabled={currentPage <= 1}
                className="px-3 py-1"
            >
                Previous
            </Button>
            <span> Page {currentPage} of {numPages} </span>
            <Button
                radius={'sm'}
                variant={"solid"}
                color={"primary"}
                onPress={() => goToPage(currentPage + 1)}
                isDisabled={currentPage >= numPages!}
                className="px-3 py-1"
            >
                Next
            </Button>
            <input
                type="number"
                min={1}
                max={numPages}
                value={currentPage}
                onChange={(e) => goToPage(Number(e.target.value))}
                className="w-20 px-2 py-1 border rounded"
            />
        </div>
    )
}