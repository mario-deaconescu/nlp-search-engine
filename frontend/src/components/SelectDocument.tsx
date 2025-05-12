import {FaPlus} from "react-icons/fa";
import {useCallback} from "react";
import {Spinner} from "@heroui/react";

export const SelectDocument = ({handleClick, handleDrop, setDragOver, dragOver, loading} : {
    handleClick: () => void,
    handleDrop: (e: any) => void,
    setDragOver: (dragOver: boolean) => void,
    dragOver: boolean,
    loading: boolean,
}) => {
    const onClick = useCallback(() => {
        if(loading) return;
        handleClick();
    }, [loading, handleClick]);

    const onDrop = useCallback((e: any) => {
        e.preventDefault();
        if(loading) return;
        handleDrop(e);
    }, [loading, handleDrop]);

    const onDragOver = useCallback((e: any) => {
        e.preventDefault();
        if(loading) return;
        setDragOver(true);
    }, [loading, setDragOver]);

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
            {loading && <Spinner/>}
        </div>
    );
}
