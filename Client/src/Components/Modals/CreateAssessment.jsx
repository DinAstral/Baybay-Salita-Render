import { useState, useEffect } from "react";
import axios from "axios";
import toast from "react-hot-toast";
import PropTypes from "prop-types";
import {
  Modal,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter,
  Button,
  Select,
  SelectItem,
} from "@nextui-org/react";

const CreateSuccess = ({ show, onHide }) => {
  const handleSuccessClick = () => {
    // Simply close the success modal without reloading the page.
    onHide();
  };
  return (
    <Modal
      isOpen={show}
      onClose={onHide}
      aria-labelledby="contained-modal-title-vcenter"
      isDismissable={false}
      isKeyboardDismissDisabled={true}
    >
      <ModalContent>
        <ModalHeader id="contained-modal-title-vcenter">
          Activity Created Successfully!
        </ModalHeader>
        <ModalBody>
          <p>You have created an activity for your students!</p>
        </ModalBody>
        <ModalFooter>
          <Button color="primary" onClick={handleSuccessClick}>
            Close
          </Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
};

const CreateAssessment = ({ show, handleClose, userId, section }) => {
  const [modalShow, setModalShow] = useState(false);
  const [words, setWords] = useState([]);
  const [sentences, setSentences] = useState([]);
  const [filteredWords, setFilteredWords] = useState([]);
  const [existingAssessments, setExistingAssessments] = useState([]);
  const [data, setData] = useState({
    Period: "",
    Type: "",
    Title: "",
    Item1: "",
    Item2: "",
    Item3: "",
    Item4: "",
    Item5: "",
    Item6: "",
    Item7: "",
    Item8: "",
    Item9: "",
    Item10: "",
  });
  const [errors, setErrors] = useState({});

  const fetchImportWord = () => {
    axios
      .get("/api/getImportWord")
      .then((response) => {
        setWords(response.data);
      })
      .catch((err) => console.log(err));
  };

  const fetchImportSentence = () => {
    axios
      .get("/api/getSentence")
      .then((response) => {
        setSentences(response.data);
      })
      .catch((err) => console.log(err));
  };

  // Fetch assessments for the given section and user.
  const fetchExistingAssessments = () => {
    axios
      .get(`/api/getAssessments?section=${section}&userId=${userId}`)
      .then((response) => {
        setExistingAssessments(response.data);
      })
      .catch((err) => console.log(err));
  };

  useEffect(() => {
    fetchImportWord();
    fetchImportSentence();
    fetchExistingAssessments();
  }, []);

  useEffect(() => {
    if (data.Type) {
      const filtered = words.filter((word) => word.Type === data.Type);
      setFilteredWords(filtered);
    } else {
      setFilteredWords([]);
    }
  }, [data.Type, words]);

  const handleWordChange = (itemKey, selectedWord) => {
    const selectedItem = filteredWords.find(
      (word) => word.ItemCode === selectedWord
    );
    setData({ ...data, [itemKey]: selectedItem ? selectedItem.ItemCode : "" });
  };

  // Assessment options defined in sequential order.
  const assessmentOptions = [
    { key: "Pagbabaybay", label: "Assessment 1: Pagbabaybay Tunog at Letra" },
    { key: "Pantig", label: "Assessment 2: Pantig" },
    { key: "Salita", label: "Assessment 3: Salita" },
    { key: "Pagbabasa", label: "Assessment 4: Pagbabasa" },
  ];

  // The next assessment that can be created for this user.
  const getNextAssessmentNumber = () => {
    return existingAssessments.length + 1;
  };

  const validateInputs = () => {
    let isValid = true;
    let newErrors = {};

    if (!data.Period) {
      newErrors.Period = "Grading Period is required.";
      isValid = false;
    }
    if (!data.Type) {
      newErrors.Type = "Type of Assessment is required.";
      isValid = false;
    }
    if (data.Type === "Pagbabasa" && !data.Title) {
      newErrors.Title = "Title is required for Pagbabasa.";
      isValid = false;
    }
    for (let i = 1; i <= 10; i++) {
      if (data.Type !== "Pagbabasa" && !data[`Item${i}`]) {
        newErrors[`Item${i}`] = `Item ${i} is required.`;
        isValid = false;
      }
    }
    setErrors(newErrors);
    return isValid;
  };

  const createAct = async (e) => {
    e.preventDefault();

    // Ensure that the selected assessment matches the next sequential unlocked assessment.
    const nextAssessment = assessmentOptions[getNextAssessmentNumber() - 1];
    if (data.Type !== nextAssessment.key) {
      toast.error("You must create assessments in sequential order.");
      return;
    }

    if (!validateInputs()) {
      toast.error("Please fill out all required fields.");
      return;
    }

    try {
      const response = await axios.post("/api/submitAssessment", {
        UserID: userId,
        Section: section,
        Period: data.Period,
        Type: data.Type,
        Title: data.Title,
        Item1: data.Item1,
        Item2: data.Item2,
        Item3: data.Item3,
        Item4: data.Item4,
        Item5: data.Item5,
        Item6: data.Item6,
        Item7: data.Item7,
        Item8: data.Item8,
        Item9: data.Item9,
        Item10: data.Item10,
      });
      if (response.data.error) {
        toast.error(response.data.error);
      } else {
        // Re-fetch assessments to unlock the next one.
        fetchExistingAssessments();
        setData({
          Period: "",
          Type: "",
          Title: "",
          Item1: "",
          Item2: "",
          Item3: "",
          Item4: "",
          Item5: "",
          Item6: "",
          Item7: "",
          Item8: "",
          Item9: "",
          Item10: "",
        });
        toast.success("Created Activity Successfully.");
        setModalShow(true);
        handleClose();
      }
    } catch (error) {
      console.log(error);
      toast.error("Failed to create the activity. Please try again.");
    }
  };

  return (
    <>
      <Modal isOpen={show} onClose={handleClose} size="4xl">
        <ModalContent>
          <ModalHeader>Create New Activity</ModalHeader>
          <form onSubmit={createAct}>
            <ModalBody>
              <Select
                label="Grading Period"
                placeholder="Select Grading Period"
                value={data.Period}
                onChange={(e) =>
                  setData({ ...data, Period: e.target.value })
                }
                isInvalid={!!errors.Period}
                errorMessage={errors.Period}
              >
                <SelectItem key="1" value="Grading Period 1">
                  Grading Period 1
                </SelectItem>
                <SelectItem key="2" value="Grading Period 2">
                  Grading Period 2
                </SelectItem>
                <SelectItem key="3" value="Grading Period 3">
                  Grading Period 3
                </SelectItem>
                <SelectItem key="4" value="Grading Period 4">
                  Grading Period 4
                </SelectItem>
              </Select>

              <Select
                label="Type of Assessment"
                placeholder="Select Type of Assessment:"
                value={data.Type}
                onChange={(e) =>
                  setData({ ...data, Type: e.target.value })
                }
                isInvalid={!!errors.Type}
                errorMessage={errors.Type}
              >
                {assessmentOptions.map(
                  (option, index) =>
                    // Only show the next unlocked assessment for creation.
                    index + 1 === getNextAssessmentNumber() && (
                      <SelectItem key={option.key} value={option.key}>
                        {option.label}
                      </SelectItem>
                    )
                )}
              </Select>

              {data.Type === "Pagbabasa" ? (
                <Select
                  label="Title"
                  value={data.Title}
                  placeholder="Select a Title:"
                  onChange={(e) =>
                    setData({ ...data, Title: e.target.value })
                  }
                  isInvalid={!!errors.Title}
                  errorMessage={errors.Title}
                >
                  {sentences.map((sentence) => (
                    <SelectItem key={sentence.Title} value={sentence.Title}>
                      {sentence.Title}
                    </SelectItem>
                  ))}
                </Select>
              ) : (
                <div className="grid grid-cols-2 gap-4">
                  {[...Array(10)].map((_, i) => (
                    <Select
                      key={`Item${i + 1}`}
                      label={`Item ${i + 1}`}
                      value={data[`Item${i + 1}`]}
                      placeholder="Select a word:"
                      onChange={(e) =>
                        handleWordChange(`Item${i + 1}`, e.target.value)
                      }
                      isInvalid={!!errors[`Item${i + 1}`]}
                      errorMessage={errors[`Item${i + 1}`]}
                    >
                      {filteredWords.map((word) => (
                        <SelectItem key={word.ItemCode} value={word.ItemCode}>
                          {word.Word}
                        </SelectItem>
                      ))}
                    </Select>
                  ))}
                </div>
              )}
            </ModalBody>
            <ModalFooter>
              <Button color="danger" onClick={handleClose}>
                Close
              </Button>
              <Button color="primary" type="submit">
                Create Activity
              </Button>
            </ModalFooter>
          </form>
        </ModalContent>
      </Modal>
      <CreateSuccess show={modalShow} onHide={() => setModalShow(false)} />
    </>
  );
};

CreateAssessment.propTypes = {
  show: PropTypes.bool.isRequired,
  handleClose: PropTypes.func.isRequired,
};

export default CreateAssessment;
