import PropTypes from "prop-types"; // Import PropTypes
import {
  Modal,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter,
  Button,
} from "@nextui-org/react";

const PrintRecord = ({ show, onHide, print }) => {
  return (
    <Modal
      isOpen={show}
      onClose={onHide}
      aria-labelledby="print-user-modal-title"
      isDismissable={false}
      isKeyboardDismissDisabled={true}
    >
      <ModalContent>
        <ModalHeader className="flex flex-col" id="print-user-modal-title">
          <h1>Print User Information</h1>
        </ModalHeader>
        <ModalBody>
          <p>
            You are now going to generate a excel file of this data. Do you want
            to continue?
          </p>
        </ModalBody>
        <ModalFooter>
          <Button variant="light" color="danger" onClick={onHide}>
            Cancel
          </Button>
          <Button color="primary" onClick={print}>
            Print
          </Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
};

// Add prop types validation
PrintRecord.propTypes = {
  show: PropTypes.bool.isRequired,
  onHide: PropTypes.func.isRequired,
  print: PropTypes.func.isRequired, // Optional prop, you can add more validation if needed
};

export default PrintRecord;
