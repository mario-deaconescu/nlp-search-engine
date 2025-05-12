import {useEffect, useMemo, useRef, useState} from "react";
import {Document, Page, pdfjs} from "react-pdf";
import "react-pdf/dist/esm/Page/AnnotationLayer.css";
import "react-pdf/dist/esm/Page/TextLayer.css";
import {FaSearch} from "react-icons/fa";
import {addToast, Button, Divider, Input, Listbox, ListboxItem, Progress, Select, SelectItem} from "@heroui/react";
import {SelectDocument} from "./components/SelectDocument.tsx";
import {PaginationButtons} from "./components/PaginationButtons.tsx";

// Set worker source for pdf.js
pdfjs.GlobalWorkerOptions.workerSrc = new URL(
    'pdfjs-dist/build/pdf.worker.min.mjs',
    import.meta.url,
).toString();

type SearchResults = {
    current: number,
    total: number,
    results: {
        page: number,
        score: number,
    }[]
}

function App() {
    const fileInputRef = useRef<HTMLInputElement>(null);
    const [pdfFile, setPdfFile] = useState<File>();
    const [dragOver, setDragOver] = useState(false);
    const [numPages, setNumPages] = useState<number>();
    const [currentPage, setCurrentPage] = useState(1);
    const [searchInput, setSearchInput] = useState("");
    const [searchTypes, setSearchTypes] = useState<Set<"TF-IDF"| "FAISS" | "BM-25">>(new Set(["TF-IDF"]));
    const [loading, setLoading] = useState(false);
    const [searchResults, setSearchResults] = useState<SearchResults | null>(null);
    const [session_id, setSessionId] = useState<string>();
    const [numResults, setNumResults] = useState(5);

    const progress = useMemo(() => {
        if (!loading) return null;
        if (!searchResults) return 0;
        return (searchResults.current + 1) / searchResults.total * 100;
    }, [loading, searchResults]);

    const filteredSearchResults = useMemo(() => {
        if (!searchResults) return null;
        return searchResults.results
            .slice(0, numResults)
            .filter((result) => {
                return result.score > 0;
            }).map((result, index) => ({
                page: result.page,
                score: result.score,
                first: index === 0,
            }));

    }, [searchResults, numResults]);

    const handleClick = () => fileInputRef.current?.click();

    const handleFiles = async (files: FileList | null) => {
        console.log("files", files);
        if (!files) return;
        const file = files[0];
        if (file && file.type === "application/pdf") {
            setPdfFile(file);
            setCurrentPage(1);
            setSearchInput("");
            setSearchResults(null);
            setSessionId(undefined);

            const formData = new FormData();
            formData.append("file", file);

            const uploadRes = await fetch("http://localhost:8000/upload", {
                method: "POST",
                body: formData,
            });

            const {session_id} = await uploadRes.json();
            setSessionId(session_id);
        } else {
            addToast({
                title: "Only PDF files are supported.",
                severity: "danger",
            })
        }
    };

    const handleSearch = async () => {

        const trimmedSearch = searchInput.trim();
        if (!pdfFile || !trimmedSearch) return;
        setLoading(true);
        setSearchResults(null);

        try {
            let eventSource;
            if(searchTypes.has("TF-IDF"))
                eventSource = new EventSource(`http://localhost:8000/search-tf-idf?session_id=${session_id}&search=${trimmedSearch}`);
            else if(searchTypes.has("FAISS"))
                eventSource = new EventSource(`http://localhost:8000/search-faiss?session_id=${session_id}&search=${trimmedSearch}`);
            else if(searchTypes.has("BM-25"))
                eventSource = new EventSource(`http://localhost:8000/search-bm25?session_id=${session_id}&search=${trimmedSearch}`);


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
        } catch (error: any) {
            console.error("Error sending PDF:", error);
            addToast({
                title: "Error sending PDF",
                severity: "danger",
                description: error.message,
            });
            throw error;

        }
    }

    const handleDrop = async (e: any) => {
        e.preventDefault();
        setDragOver(false);
        await handleFiles(e.dataTransfer.files);
    };

    useEffect(() => {
        if (!loading && searchResults) {
            setCurrentPage(searchResults.results[0].page)
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
                <div className={"flex flex-row gap-4 h-min w-full justify-center overflow-y-scroll"}>
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
                            <p>Search type</p>
                            <Select
                                selectedKeys={searchTypes}
                                // defaultSelectedKeys={new Set(["TF-IDF"])}
                                onSelectionChange={(e) => setSearchTypes(e)}
                                aria-label="Search Type"
                                labelPlacement={"outside"}
                                color={"primary"}
                            >
                                <SelectItem key={"TF-IDF"}>TF-IDF</SelectItem>
                                <SelectItem key={"FAISS"}>FAISS</SelectItem>
                                <SelectItem key={"BM-25"}>BM-25</SelectItem>

                            </Select>
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
                                    isLoading={loading}
                                    color={"primary"}>Search</Button>
                            </div>
                            {progress !== null &&
                                <Progress aria-label="Loading..." className="max-w-md" value={progress}/>}
                        </div>
                        <div className={"flex flex-col w-full gap-6 self-center"}>
                            {/*{loading && <Spinner/>}*/}
                            {filteredSearchResults &&
                                <div>
                                    <p className={'text-xl'}>
                                        Top Results:
                                    </p>
                                    <Listbox aria-label="Search Results" items={filteredSearchResults}
                                             onAction={(key) => setCurrentPage((typeof key === "string" ? parseInt(key) : key))}>
                                        {(item) => (
                                            <ListboxItem
                                                key={item.page}
                                                // className={item.key === "delete" ? "text-danger" : ""}
                                                color={item.first ? "primary" : "default"}
                                            >
                                                <div>
                                                    <p>Page {item.page}</p>
                                                    <p className={'text-xs'}>Score: {item.score}</p>
                                                </div>
                                            </ListboxItem>
                                        )}
                                    </Listbox>
                                </div>
                            }
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
    )
        ;
}

export default App;

