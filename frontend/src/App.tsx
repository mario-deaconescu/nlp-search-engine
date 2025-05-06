import {useEffect, useRef, useState} from "react";
import {Page, pdfjs, Document} from "react-pdf";
import "react-pdf/dist/esm/Page/AnnotationLayer.css";
import "react-pdf/dist/esm/Page/TextLayer.css";
import {FaPlus, FaSearch} from "react-icons/fa";
import {addToast, Button, Divider, Input, Spinner} from "@heroui/react";
import {SelectDocument} from "./components/SelectDocument.tsx";
import {PaginationButtons} from "./components/PaginationButtons.tsx";

// Set worker source for pdf.js
pdfjs.GlobalWorkerOptions.workerSrc = new URL(
    'pdfjs-dist/build/pdf.worker.min.mjs',
    import.meta.url,
).toString();

function App() {
    const fileInputRef = useRef(null);
    const [pdfFile, setPdfFile] = useState<File>();
    const [dragOver, setDragOver] = useState(false);
    const [numPages, setNumPages] = useState<number>();
    const [currentPage, setCurrentPage] = useState(1);
    const [searchInput, setSearchInput] = useState("");
    const [loading, setLoading] = useState(false);
    const [searchResults, setSearchResults] = useState<{page:number,score:number}>();
    const [session_id, setSessionId] = useState<string>();

    const handleClick = () => fileInputRef.current.click();

    const handleFiles = async (files: FileList | null) => {
        console.log("files", files);
        if (!files) return;
        const file = files[0];
        if (file && file.type === "application/pdf") {
            setPdfFile(file);
            setCurrentPage(1);
            setSearchInput("");

            const formData = new FormData();
            formData.append("file", file);

            const uploadRes = await fetch("http://localhost:8000/upload", {
                method: "POST",
                body: formData,
            });

            const { session_id } = await uploadRes.json();
            setSessionId(session_id);
        } else {
            alert("Only PDF files are supported.");
        }
    };

    const handleSearch = async () => {

        const trimmedSearch = searchInput.trim();
        if (!pdfFile || !trimmedSearch ) return;
        setLoading(true);
        setSearchResults(undefined);

        try {
            const eventSource = new EventSource(`http://localhost:8000/search-stream?session_id=${session_id}&search=${trimmedSearch}`);

            eventSource.onmessage = (event) => {
                const data = JSON.parse(event.data);
                console.log("SSE data:", data);
                setSearchResults(data);
            };

            eventSource.onerror = (err) => {
                console.error("SSE error:", err);
                eventSource.close();
                setLoading(false);
            };

            eventSource.onopen = () => console.log("Connected to SSE");
        } catch (error) {
            console.error("Error sending PDF:", error);
            throw error;
        }
    }

    const handleDrop = async (e) => {
        e.preventDefault();
        setDragOver(false);
        await handleFiles(e.dataTransfer.files);
    };

    useEffect(() => {
        console.log("pdfFile", pdfFile);
    }, [pdfFile]);

    useEffect(() => {
        if(!loading && searchResults){
            setCurrentPage(searchResults.page + 1)
        }
    }, [loading, searchResults]);

    return (
        <div className="h-screen max-h-screen flex flex-col items-center justify-center bg-gray-50 p-5">
            {!pdfFile &&
                <SelectDocument
                    handleDrop={handleDrop}
                    handleClick={handleClick}
                    setDragOver={setDragOver}
                    dragOver={dragOver}
                />
            }
            <input
                type="file"
                accept="application/pdf"
                multiple={false}
                ref={fileInputRef}
                className="hidden"
                onChange={(e) => handleFiles(e.target.files)}
            />

            {pdfFile && (
                <div className={"flex flex-row gap-4 h-min w-full justify-center"}>
                    <div
                        className="w-[70%] max-w-3xl h-fit max-h-full bg-white p-4 rounded shadow-lg flex flex-col items-center">
                        <Document
                            file={pdfFile}
                            onLoadSuccess={({numPages}) => setNumPages(numPages)}
                        >
                            <Page
                                pageNumber={currentPage}
                                renderAnnotationLayer={false}
                                renderTextLayer={false}
                            />
                        </Document>
                        <Divider/>
                        <PaginationButtons
                            currentPage={currentPage}
                            setCurrentPage={setCurrentPage}
                            numPages={numPages!}
                        />

                    </div>
                    <div
                        className={"p-4 flex flex-col items-start gap-4 justify-between h-full "}>
                        <div className={"flex flex-col w-full gap-2"}>
                            <p>Search in document</p>
                            <div className={"flex flex-row gap-4 w-full"}>
                            <Input
                                startContent={<FaSearch/>}
                                placeholder="Search"
                                color={"primary"}
                                type="search"
                                value={searchInput}
                                onChange={(e) => setSearchInput(e.target.value)}
                                className="w-full"
                            />
                                <Button
                                    onPress={handleSearch}
                                    variant={"bordered"}
                                color={"primary"}>Search</Button>
                            </div>
                        </div>
                        <div className={"flex flex-col w-full gap-6 self-center"}>
                            {loading && <Spinner/>}
                            {searchResults && <div className={loading ? "text-gray-500" : ""}>
                                <p>Your answer is on page: {searchResults.page + 1}</p>
                                <p>Score: {searchResults.score}</p>
                            </div>}
                        </div>
                        <div className={"flex flex-row gap-4"}>
                            <Button
                                color={"primary"}
                                onPress={() => handleClick()}>
                                New Document
                            </Button>
                            <Button
                                variant={"bordered"}
                                color={"danger"}
                                onPress={() => setPdfFile(undefined)}
                            >
                                Remove Document
                            </Button>
                        </div>
                    </div>
                </div>
            )}

        </div>
    );
}

export default App;

