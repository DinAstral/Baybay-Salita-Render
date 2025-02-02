import React from "react";
import ContentHeader from "../../Components/ContentDasboard/ContentHeader";
import ParentSidebar from "../../Components/ParentDashboardComponent/ParentSidebar";
import BodyInformationKid from "../../Components/ParentDashboardComponent/BodyInformationKid";

const InformationKid = () => {
  return (
    <div className="w-full h-full flex">
      {/* Sidebar */}
      <div className="fixed top-0 left-0 w-[15%] h-full bg-[#F5EBD6] shadow-lg">
        <ParentSidebar />
      </div>

      {/* Main Dashboard Content */}
      <div className="ml-[15%] w-[85%] h-full min-h-[100vh] bg-[#F2EDE3] p-[2rem] flex flex-col gap-[1.5rem]">
        <ContentHeader />
        <BodyInformationKid />
      </div>
    </div>
  );
};

export default InformationKid;
