import {FaPlus} from "react-icons/fa";

export const SelectDocument = ({handleClick, handleDrop, setDragOver, dragOver} : {
    handleClick: () => void,
    handleDrop: (e: any) => void,
    setDragOver: (dragOver: boolean) => void,
    dragOver: boolean,
}) => {
    return (
        <div
            onClick={handleClick}
            onDrop={handleDrop}
            onDragOver={(e) => {
                e.preventDefault();
                setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            className={`w-full max-w-xl h-64 border-4 border-dashed rounded-2xl flex flex-col items-center justify-center cursor-pointer transition-colors hover:border-blue-500 hover:bg-blue-50 ${
                dragOver ? "border-blue-500 bg-blue-50" : "border-gray-300 bg-white"
            }`}
        >
            <FaPlus className="w-12 h-12 text-gray-500 mb-2"/>
            <p className="text-gray-700 font-medium text-lg">Click or drag PDF files here to upload</p>
        </div>
    );
}